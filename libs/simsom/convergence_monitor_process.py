"""
Reponsible for monitoring the convergence of the simulation.
Send termination signal to all processes when the simulation has converged.
"""

import time
from glob import glob
from collections import Counter
import csv
import numpy as np
from mpi4py import MPI


def obtain_diversity(messages_id: list):
    """Get diversity value based on shared messages.
    The function take in input the list of all messages
    that has been created and calculate diversity

    Args:
        messages_id (list): list of messages id

    Returns:
        float: diversity value
    """
    humanshares = []
    for message_id in messages_id:
        humanshares += [message_id]
    message_counts = Counter(humanshares)
    count_byid = sorted(dict(message_counts).items())
    humanshares = np.array([m[1] for m in count_byid])
    hshare_pct = np.divide(humanshares, sum(humanshares))
    diversity = np.sum(hshare_pct * np.log(hshare_pct)) * -1
    return diversity


def run_convergence_monitor(
    comm_world: MPI.Intercomm,
    rank: int,
    rank_index: dict,
    sliding_window_convergence: int,
    message_count_target: int,
    convergence_param: float,
    verbose: bool,
    print_interval: int,
):
    """
    Function that takes care of calculating the convergence condition and stop execution

    Args:
        comm (MPI.COMM_WORLD): communication context between processes
        sliding_window_convergence (int): sliding window size for quality analysis
        FILE_PATH (str): path to the file where the activities are saved
    """

    # Status of the processes
    # status = MPI.Status()

    # Bootstrap sync
    comm_world.Barrier()

    # Files paths
    file_paths = glob("files/*/activities.csv")
    file_paths.sort()

    # Take the most recent file
    file_path = file_paths[-1]

    # It contains a list of quality values, used to calculate quality between windows
    current_quality = []
    current_appeal = []
    # Number of rows we have, used to clean the file and manage max_iter
    count_index = 0
    # Count number of iterations to create intervals for print
    count_interval = 0
    # Count number of iterations to create windows
    current_window = 0
    # Keep track of overall quality and appeal
    overall_quality = []
    overall_appeal = []
    # First window shouldn't be considered
    previous_avg = None
    # For diversity
    message_id_list = []

    # Read the activity file, where we store the messages that has been created
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        # Make sure that the activity file is populated (so the simulation is running)
        while True:
            line = csvfile.readline()
            if (len(line) == 0) or (not line):
                while True:
                    line = csvfile.readline()
                    time.sleep(0.01)
                    if len(line) > 0:
                        break

            # Get a message
            row = line.strip().split(",")
            count_index += 1

            # Obtains data related to the activities
            message_id = row[0]
            # user_id = row[1]
            quality = float(row[2])
            appeal = float(row[3])

            # Get quality to print quality interval
            current_window += 1
            count_interval += 1
            # Get the quality for the sliding window difference and appeal
            current_quality.append(quality)
            current_appeal.append(appeal)

            # Get the message id to calculate diversity
            if message_id.startswith("P"):
                message_id_list.append(message_id)
            else:
                message_id_list.append(row[6])

            # Get the data for the entire period
            overall_appeal.append(appeal)
            overall_quality.append(quality)

            if verbose:
                # If we want to see info during the execution, based on n. of activities
                if count_interval == print_interval:
                    # Print the average quality for the interval
                    print(f"- Average quality: {np.mean(current_quality)}")
                    # Print the average appeal for the interval
                    print(f"- Average appeal: {np.mean(current_appeal)}")
                    # Print the average diversity for the interval
                    print(f"- Average diversity: {obtain_diversity(message_id_list)}")
                    count_interval = 0
                    print("---------------------")

            # When we reach the sliding window size we calculate the current average
            if current_window == sliding_window_convergence:
                current_avg = np.nanmean(current_quality)

                # We check if it is not the first window and compare with the window before
                if previous_avg:
                    diff = abs(current_avg - previous_avg)
                    if verbose:
                        print("Window end")
                        print(f"- Quality difference between windows: {diff}")
                        print("---------------------")

                    # Check if the execution is managed by convergence between windows and check the termination param
                    if message_count_target == 0:
                        if diff <= convergence_param:
                            data = np.array([count_index], dtype="i")
                            req = comm_world.Isend(
                                data, dest=rank_index["data_manager"]
                            )  # non blocking send
                            req.Wait()
                            if verbose:
                                print(
                                    "-- Overall average quality: ",
                                    np.nanmean(overall_quality),
                                )
                                print(
                                    "-- Overall average appeal: ",
                                    np.nanmean(overall_appeal),
                                )
                                print(
                                    "-- Overall average diversity: ",
                                    obtain_diversity(message_id_list),
                                )
                            break
                    # Otherwise send stats without convergence monitor
                    elif count_index >= message_count_target:
                        if verbose:
                            print(
                                "-- Overall average quality: ",
                                np.nanmean(overall_quality),
                            )
                            print(
                                "-- Overall average appeal: ",
                                np.nanmean(overall_appeal),
                            )
                            print(
                                "-- Overall average diversity: ",
                                obtain_diversity(message_id_list),
                            )
                        break

                previous_avg = current_avg
                current_window = 0
                current_quality = []
