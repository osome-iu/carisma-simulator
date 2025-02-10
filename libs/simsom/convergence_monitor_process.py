"""
Reponsible for monitoring the convergence of the simulation.
Send termination signal to all processes when the simulation has converged.
"""

import time
from glob import glob
import pandas as pd
import csv
import numpy as np
from mpi4py import MPI


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
    diversity_file_paths = glob("files/*/diversity_log.txt")
    diversity_file_paths.sort()
    file_paths = glob("files/*/activities.csv")
    file_paths.sort()

    # Take the most recent file
    diversity_file_path = diversity_file_paths[-1]
    file_path = file_paths[-1]

    previous_window = []
    current_window = []
    current_sum = 0
    count_index = 0
    check_print = []
    overall_quality = []
    overall_appeal = []

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
            quality = float(row[2])
            appeal = float(row[3])

            # Get quality to print quality interval
            check_print.append(quality)
            current_window.append(quality)

            # Get the quality for the sliding window difference
            current_sum += quality

            # Get the data for the entire period
            overall_appeal.append(appeal)
            overall_quality.append(quality)

            if verbose:
                # If we want to see info during the execution, based on n. of activities
                if len(check_print) == print_interval:
                    # Print the average quality for the interval
                    print(f"- Average quality: {np.mean(check_print)}")
                    with open(
                        diversity_file_path, "r", encoding="utf-8"
                    ) as diversity_file:
                        diversity_value = diversity_file.readline().strip()
                    # Print the average appeal for the interval
                    print(f"- Average diversity: {diversity_value}")
                    check_print = []
                    print("---------------------")

            # When we reach the sliding window size we calculate the current average
            if len(current_window) == sliding_window_convergence:
                current_avg = current_sum / sliding_window_convergence

                # We check if it is not the first window and compare with the window before
                if previous_window:
                    previous_avg = sum(previous_window) / sliding_window_convergence
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
                        break

                previous_window = current_window
                current_window = []
                current_sum = 0
