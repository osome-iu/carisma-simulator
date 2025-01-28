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
):
    """
    Function that takes care of calculating the convergence condition and stop execution

    Args:
        comm (MPI.COMM_WORLD): communication context between processes
        sliding_window_convergence (int): sliding window size for quality analysis
        FILE_PATH (str): path to the file where the activities are saved
    """
    print(f"Convergence monitor start @ rank: {rank}")

    # Status of the processes
    # status = MPI.Status()

    # Bootstrap sync
    comm_world.Barrier()

    file_paths = glob("files/*/activities.csv")
    file_paths.sort()
    file_path = file_paths[-1]
    previous_window = []
    current_window = []
    current_sum = 0
    count_index = 0
    overall_appeal = []

    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        while True:
            line = csvfile.readline()
            if (len(line) == 0) or (not line):
                while True:
                    line = csvfile.readline()
                    time.sleep(0.01)
                    if len(line) > 0:
                        break
            row = line.strip().split(",")
            count_index += 1
            quality = float(row[2])
            current_window.append(quality)
            current_sum += quality
            overall_appeal.append(float(row[3]))
            if verbose:
                if len(current_window) == 250:
                    print(f"- Average quality: {current_sum / 250}")
            if len(current_window) == sliding_window_convergence:
                current_avg = current_sum / sliding_window_convergence

                if previous_window:
                    previous_avg = sum(previous_window) / sliding_window_convergence
                    diff = abs(current_avg - previous_avg)
                    if verbose:
                        print(f"- Quality difference between windows: {diff}")

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
                                    current_sum / count_index,
                                )
                                print(
                                    "-- Overall average appeal: ",
                                    np.nanmean(overall_appeal),
                                )
                            break
                    elif count_index >= message_count_target:
                        if verbose:
                            print(
                                "-- Overall average quality: ",
                                current_sum / count_index,
                            )
                            print(
                                "-- Overall average appeal: ",
                                np.nanmean(overall_appeal),
                            )
                        break

                previous_window = current_window
                current_window = []
                current_sum = 0

    print(f"Convergence monitor stop @ rank: {rank}")
