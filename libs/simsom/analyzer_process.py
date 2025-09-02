"""
Reponsible for monitoring the convergence of the simulation.
Send termination signal to all processes when the simulation has converged.
"""

import time
import csv
import numpy as np
from collections import Counter
from mpi4py import MPI
import simtools
import pandas as pd
from mpi_utils import iprobe_with_timeout, clean_termination, handle_crash, gettimestamp
import os

# Path files
time_now = int(time.time())
folder_path = f"files/{time_now}"
file_path_activity = folder_path + "/activities.csv"
file_path_passivity = folder_path + "/passivities.csv"
rho = 0.8


def resize_output(size: int):
    """Resize output file to make sure we do not
        persist data that has been created after the
        interrupt signal from convergence monitor.
        The function read and remove the part of the file
        that we shouldn't have because of send / receive delays

    Args:
        size (int): size of the file that we should have
    """
    df = pd.read_csv(file_path_activity)
    df = df[:size]
    df.to_csv(
        file_path_activity,
        lineterminator="\n",
        index=False,
        encoding="utf-8",
    )

    # get last user id from activity file so we can clean also the passivity file
    last_user = df.tail(1).user_id.values[0]
    df = pd.read_csv(file_path_passivity)
    last_idx = df[df["user_id"] == last_user].last_valid_index()
    df = df.loc[:last_idx]
    df.to_csv(
        file_path_passivity,
        lineterminator="\n",
        index=False,
        encoding="utf-8",
    )


def update_quality(current_quality, overall_avg_quality):
    """
    Update quality using exponential moving average to ensure stable state at convergence
    Forget the past slowly, i.e., new_quality = 0.8 * avg_quality(at t-1) + 0.2 * current_quality
    """

    new_quality = rho * current_quality + (1 - rho) * overall_avg_quality
    quality_diff = (
        abs(new_quality - current_quality) / current_quality
        if current_quality > 0
        else 0
    )
    return quality_diff, new_quality


def enforce_single_convergence_method(**methods) -> dict:
    """
    Enforce a single convergence method to be used
    If multiple methods are set to True, the first one in the list will be used

    Returns:
        dict: dictionary with the selected method
    """
    priority = [
        "day_count_criterion",
        "sliding_window_method",
        "ema_quality_method",
    ]

    active = [key for key in priority if methods.get(key, False)]
    selected = active[0] if active else priority[0]

    return {key: key == selected for key in priority}


def entropy(x):
    return np.sum(x * np.log(x))


def measure_diversity(self) -> int:
    """
    Calculates the diversity of the system using entropy (in terms of unique messages)
    (Invoke only after self._return_all_message_info() is called)
    """

    humanshares = []
    for human_id in self.users:
        newsfeed = self.users[human_id].user_feed
        message_ids, _, _ = newsfeed
        for message_id in message_ids:
            humanshares += [message_id]
    message_counts = Counter(humanshares)
    # return a list of [(messageid, count)], sorted by id
    count_byid = sorted(dict(message_counts).items())
    humanshares = np.array([m[1] for m in count_byid])

    hshare_pct = np.divide(humanshares, sum(humanshares))
    diversity = entropy(hshare_pct) * -1
    # Note that (np.sum(humanshares)+np.sum(botshares)) !=self.num_messages because a message can be shared multiple times
    return diversity


def run_analyzer(
    comm_world: MPI.Intracomm,
    rank: int,
    rank_index: dict,
    # Params for sliding window method
    sliding_window_method: bool,
    sliding_window_size: int,
    sliding_window_threshold: float,
    # Params for max target method
    day_count_criterion: bool,
    target_days: int,
    # Params for exponential moving average method
    ema_quality_method: bool,
    ema_quality_convergence: float,
    # Number of users to be used for the simulation
    n_users: int,
    # Params for printing stuff during the execution
    verbose: bool,
    print_interval: int,
    # Params for saving activities on disk
    save_active_interactions: bool = True,
    save_passive_interactions: bool = True,
):
    """
    Function that takes care of calculating the convergence condition and stop execution

    Args:
        comm (MPI.COMM_WORLD): communication context between processes
        sliding_window_convergence (int): sliding window size for quality analysis
        FILE_PATH (str): path to the file where the activities are saved
    """

    print(f"[{gettimestamp()}] Analyzer (PID: {os.getpid()}) >> running...", flush=True)

    status = MPI.Status()

    # keep track of the number of messages
    data_size = 0
    # keep track of the number of users
    intermediate_n_user = 0
    # keep track of the quality of the messages for interval printing
    interval_quality = 0
    # keep track of the quality of the messages
    quality_sum = 0
    # keep track of the number of messages for verbose debug
    message_count = 0
    # list of qualities for the sliding window
    current_quality_list = []
    # quality of the previous window
    previous_quality = None
    # dictionary of feeds for the users, this is used to calculate diversity, quality, etc.
    feeds = {}
    # list of users for the ema quality
    ema_users = []
    # value to calculate the current quality each N iterations (or after T time)
    current_quality = 1

    # Initialize files if we are saving on disk
    if save_active_interactions or save_passive_interactions:
        simtools.init_files(folder_path, file_path_activity, file_path_passivity)
    # TODO: do not create a file if is not necessary (e.g. empty passivity file)

    # Activities writings
    out_act = None
    act_writer = None
    if save_active_interactions:
        out_act = open(file_path_activity, "a", newline="", encoding="utf-8")
        act_writer = csv.writer(out_act)

    # Passivities writings
    out_pas = None
    pas_writer = None
    if save_passive_interactions:
        out_pas = open(file_path_passivity, "a", newline="", encoding="utf-8")
        pas_writer = csv.writer(out_pas)

    # Convergence mode handling
    convergence_flags = enforce_single_convergence_method(
        day_count_criterion=day_count_criterion,
        sliding_window_method=sliding_window_method,
        ema_quality_method=ema_quality_method,
    )

    day_count_criterion = convergence_flags["day_count_criterion"]
    sliding_window_method = convergence_flags["sliding_window_method"]
    ema_quality_method = convergence_flags["ema_quality_method"]

    if day_count_criterion:
        exec_name = "Max Activities"
    elif sliding_window_method:
        exec_name = "Sliding Windows"
    else:
        exec_name = "Exponential Moving Average"

    print(f"[{gettimestamp()}] Analyzer >> Execution with {exec_name}")

    alive = True

    day_count = 0

    # Bootstrap sync
    comm_world.barrier()

    while True:

        if iprobe_with_timeout(comm_world, status=status, pname="Analyzer"):

            # Receive incoming data (from any process is sending)
            sender, payload = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

            _ = sender  # not used, temporary for readability

            # Check if termination signal has been sent
            if alive and payload == "STOP":
                print(
                    f"[{gettimestamp()}] Analyzer >> stop signal detected...",
                    flush=True,
                )
                alive = False

            if alive:

                # Unpack users and passivities (visualizations)
                users, activities, passivities = payload[0]
                # NOTE: in future activities could become user newsfeeds

                # Unpack firehose chunk
                firehose_chunk = payload[1]

                # Update the number of total activities
                data_size += len(firehose_chunk)

                # Update the number of processed users
                intermediate_n_user += len(users)

                # Update global stats variables
                for a in firehose_chunk:
                    quality_sum += a.quality
                    interval_quality += a.quality
                    # NOTE: Why use a duplicate value variable? TODO: Fixit
                    message_count += 1

                    if act_writer:
                        act_writer.writerow(a.write_action())

                # # Update newsfeeds stats variables
                # for u in users:
                #     # newsfeed = u.newsfeed
                #     # avg_quality += avg_quality(newsfeed)
                #     pass
                # NOTE: for the future

                # Write the passive interactions (views)
                if pas_writer:
                    for p in passivities:
                        pas_writer.writerow(p.write_action())

                # Update elapsed days
                if firehose_chunk:
                    day_count = round(firehose_chunk[-1].time, 1)

                if verbose:
                    if message_count != 0 and intermediate_n_user % print_interval == 0:
                        print(
                            f"[{gettimestamp()}] Analyzer >> Intermediate stats after",
                            f"{intermediate_n_user} users: interval quality -->",
                            f"{round(interval_quality / message_count, 3)}",
                            f"Days: {day_count}",
                            flush=True,
                        )
                        message_count = 0
                        interval_quality = 0

                # Based on the method for convergence check if we should stop
                if day_count_criterion:

                    # Stop and terminate the process
                    if day_count >= target_days:

                        # Resize the output file to the number of messages
                        # resize_output(max_actvities_target)

                        print(
                            f"[{gettimestamp()}] Analyzer >> Average quality:",
                            round(quality_sum / data_size, 3),
                            flush=True,
                        )

                        clean_termination(
                            comm_world=comm_world,
                            sender_rank=rank,
                            sender_role="analyzer",
                            log_name="Analyzer",
                            message="GOAL REACHED, TERMINATING SIMULATION...",
                        )

                        alive = False

                # Use the convergence with sliding window or based on overall messages
                elif sliding_window_method:

                    # Save the quality of the messages in the current window
                    # current_quality_list.extend([m.quality for m in activities])  # type: ignore
                    current_quality_list.extend([m.quality for m in firehose_chunk])

                    # Calculate the average quality for this window and compare to the previous one,
                    # if the abs difference is less than the threshold break and send termination signal

                    # Check if we reached the sliding window size
                    if len(current_quality_list) >= sliding_window_size:
                        # Calculate the average quality for this window
                        current_quality = np.mean(current_quality_list)
                        # Calculate the average quality for the previous window
                        if previous_quality is not None:
                            # Check if the difference is less than the threshold
                            if (
                                abs(current_quality - previous_quality)
                                <= sliding_window_threshold
                            ):

                                print(
                                    f"[{gettimestamp()}] Analyzer >> Threshold reached:",
                                    abs(current_quality - previous_quality),
                                    flush=True,
                                )

                                print(
                                    f"[{gettimestamp()}] Analyzer >> Average quality:",
                                    round(quality_sum / data_size, 2),
                                    flush=True,
                                )

                                clean_termination(
                                    comm_world=comm_world,
                                    sender_rank=rank,
                                    sender_role="analyzer",
                                    log_name="Analyzer",
                                    message="GOAL REACHED, TERMINATING SIMULATION...",
                                )

                                alive = False

                        # Update the previous quality
                        previous_quality = current_quality
                        current_quality_list = []

                # Use the convergence with exponential moving average
                elif ema_quality_method:

                    for user in users:
                        ema_users.append(user)  # ???
                        feeds[user.uid] = user.newsfeed

                    if len(ema_users) >= n_users:
                        # NOTE: ema_users used only for len, pls implement a counter
                        ema_users = []  # ???

                        if data_size > 0:
                            # temp solution. At first stages n_data is 0.

                            print(
                                f"[{gettimestamp()}] Analyzer >> current quality {current_quality}",
                                flush=True,
                            )

                            print(
                                f"[{gettimestamp()}] Analyzer >> ndata {data_size}",
                                flush=True,
                            )

                            print(
                                f"[{gettimestamp()}] Analyzer >> avg quality {quality_sum/data_size}",
                                flush=True,
                            )

                            quality_diff, new_quality = update_quality(
                                current_quality=current_quality,
                                overall_avg_quality=quality_sum / data_size,
                            )

                            print(
                                f"[{gettimestamp()}] Analyzer >> quality diff {quality_diff}",
                                flush=True,
                            )

                            print(
                                f"[{gettimestamp()}] Analyzer >> new quality {new_quality}",
                                flush=True,
                            )

                            current_quality = new_quality

                            if quality_diff <= ema_quality_convergence:

                                print(
                                    f"[{gettimestamp()}] Analyzer >> quality_diff {quality_diff} <= threshold {ema_quality_convergence}",
                                    flush=True,
                                )

                                print(
                                    f"[{gettimestamp()}] Analyzer >> Average quality:",
                                    round(quality_sum / data_size, 2),
                                    flush=True,
                                )

                                clean_termination(
                                    comm_world=comm_world,
                                    sender_rank=rank,
                                    sender_role="analyzer",
                                    log_name="Analyzer",
                                    message="GOAL REACHED, TERMINATING SIMULATION...",
                                )

                                alive = False

                                # Resize the output file to the number of messages
                                # resize_output(max_iteration_target)

                            else:

                                print(
                                    f"[{gettimestamp()}] Analyzer >> Quality diff after {data_size} messages: {quality_diff}"
                                )

        else:

            if alive:

                handle_crash(
                    comm_world=comm_world,
                    status=status,
                    srank=rank,
                    srole="analyzer",
                    pname="Analyzer",
                )

            print(f"[{gettimestamp()}] Analyzer >> entering barrier...", flush=True)
            comm_world.barrier()
            break

    # Close writing streams
    if out_act is not None:
        out_act.close()
    if out_pas is not None:
        out_pas.close()

    print(f"[{gettimestamp()}] Analyzer >> closed.", flush=True)
