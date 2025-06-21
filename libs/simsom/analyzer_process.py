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
import random
from mpi_utils import iprobe_with_timeout

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
        "max_interactions_method",
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
    max_interactions_method: bool,
    max_iteration_target: int,
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

    print("* Analyzer process >> running...", flush=True)

    status = MPI.Status()

    # keep track of the number of messages
    n_data = 0
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
    # ????
    csv_out_act = None

    convergence_flags = enforce_single_convergence_method(
        max_interactions_method=max_interactions_method,
        sliding_window_method=sliding_window_method,
        ema_quality_method=ema_quality_method,
    )

    max_interactions_method = convergence_flags["max_interactions_method"]
    sliding_window_method = convergence_flags["sliding_window_method"]
    ema_quality_method = convergence_flags["ema_quality_method"]

    if max_interactions_method:
        exec_name = "Max iterations"
    elif sliding_window_method:
        exec_name = "Sliding windows"
    else:
        exec_name = "Exponential moving average"

    print(f"* Analyzer >> Execution with {exec_name}")

    # Initialize files
    simtools.init_files(folder_path, file_path_activity, file_path_passivity)

    # Function to terminate the process and print information
    def clean_termination_v2(rank_index: dict):
        """
        Clean termination of the process
        """

        print("* Analyzer >> GOAL REACHED, TERMINATING SIMULATION...", flush=True)

        proc_ranks = list(range(comm_world.size))
        random.shuffle(proc_ranks)
        for rank in proc_ranks:
            if rank != rank_index["analyzer"]:
                comm_world.isend("STOP", dest=rank, tag=99)
                print(
                    f"* Analyzer >> sent termination signal to rank {rank}", flush=True
                )

    alive = True

    # Bootstrap sync
    comm_world.barrier()

    while True:

        if iprobe_with_timeout(
            comm_world,
            source=MPI.ANY_SOURCE,
            tag=MPI.ANY_TAG,
            status=status,
        ):

            # Wait for data from policy filter
            data = comm_world.recv(
                source=rank_index["recommender_system"], status=status
            )

            if alive:

                # Unpack the data
                users, activities, passivities = data
                # Count the number of messages
                n_data += len(activities)
                # Count the number of user until now
                intermediate_n_user += len(users)  # previous: increment only by 1
                # print(f"n_data {n_data}", flush=True)
                # print(f"intermediate_n_user: {intermediate_n_user}", flush=True)

                # Write the data to the files
                # Write the active interactions (post/repost)
                out_act = None
                if save_active_interactions:
                    out_act = open(
                        file_path_activity,
                        "a",
                        newline="",
                        encoding="utf-8",
                    )
                    csv_out_act = csv.writer(out_act)

                try:
                    for m in activities:
                        quality_sum += m.quality
                        interval_quality += m.quality
                        message_count += 1
                        if csv_out_act:
                            csv_out_act.writerow(m.write_action())
                finally:
                    if out_act:
                        # TO FIX: out_act possibly not bound
                        out_act.close()

                # Write the passive interactions (view)
                if save_passive_interactions:
                    with open(
                        file_path_passivity, "a", newline="", encoding="utf-8"
                    ) as out_pas:
                        csv_out_pas = csv.writer(out_pas)
                        for a in passivities:
                            csv_out_pas.writerow(a.write_action())

                if verbose:
                    if message_count != 0 and intermediate_n_user % print_interval == 0:
                        print(
                            f"* Analyzer >> Intermediate stats after {intermediate_n_user} users: interval quality --> {round(interval_quality / message_count, 2)}",
                            flush=True,
                        )
                        message_count = 0
                        interval_quality = 0

                # Based on the method for convergence check if we should stop
                if max_interactions_method:

                    # Stop and terminate the process
                    if n_data >= max_iteration_target:

                        clean_termination_v2(rank_index=rank_index)

                        # Resize the output file to the number of messages
                        resize_output(max_iteration_target)

                        print(
                            "* Analyzer >> Average quality:",
                            round(quality_sum / n_data, 2),
                            flush=True,
                        )

                        alive = False

                # Use the convergence with sliding window or based on overleall messages
                elif sliding_window_method:

                    # Save the quality of the messages in the current window
                    current_quality_list.extend([m.quality for m in activities])

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
                                clean_termination_v2(rank_index=rank_index)
                                # Resize the output file to the number of messages
                                # resize_output(n_data)
                                print(
                                    "* Analyzer >> Threshold reached:",
                                    abs(current_quality - previous_quality),
                                    flush=True,
                                )

                                print(
                                    "* Analyzer >> Average quality:",
                                    round(quality_sum / n_data, 2),
                                    flush=True,
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

                        if n_data > 0:
                            # temp solution. At first stages n_data is 0.

                            print(
                                f"* Analyzer >> current quality {current_quality}",
                                flush=True,
                            )
                            print(
                                f"* Analyzer >> quality sum {quality_sum}", flush=True
                            )
                            print(f"* Analyzer >> ndata {n_data}", flush=True)
                            print(
                                f"* Analyzer >> avg quality {quality_sum/n_data}",
                                flush=True,
                            )
                            quality_diff, new_quality = update_quality(
                                current_quality=current_quality,
                                overall_avg_quality=quality_sum / n_data,
                            )
                            print(
                                f"* Analyzer >> quality diff {quality_diff}", flush=True
                            )
                            print(
                                f"* Analyzer >> new quality {new_quality}", flush=True
                            )

                            current_quality = new_quality

                            if quality_diff <= ema_quality_convergence:

                                print(
                                    f"* Analyzer >> quality_diff {quality_diff} <= threshold {ema_quality_convergence}",
                                    flush=True,
                                )

                                clean_termination_v2(rank_index=rank_index)

                                # Resize the output file to the number of messages
                                # resize_output(max_iteration_target)

                                print(
                                    "* Analyzer >> Average quality:",
                                    round(quality_sum / n_data, 2),
                                    flush=True,
                                )

                                alive = False

                            else:

                                print(
                                    f"* Analyzer >> Quality diff after {n_data} messages: {quality_diff}"
                                )

        else:

            print("* Analyzer >> entering barrier...", flush=True)
            comm_world.barrier()
            break

    print("* Analyzer >> closed.", flush=True)
