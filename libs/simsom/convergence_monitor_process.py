"""
Reponsible for monitoring the convergence of the simulation.
Send termination signal to all processes when the simulation has converged.
"""

import time
from glob import glob
import pandas as pd
import numpy as np
from mpi4py import MPI


def run_convergence_monitor(
    comm_world: MPI.Intercomm,
    rank: int,
    rank_index: dict,
    threshold: int,
    message_count_target: int,
    convergence_param: float,
):
    """
    Function that takes care of calculating the convergence condition and stop execution

    Args:
        comm (MPI.COMM_WORLD): communication context between processes
        threshold (int): sliding window size for quality analysis
        FILE_PATH (str): path to the file where the activities are saved
    """
    print("Convergence monitor start")

    # Status of the processes
    # status = MPI.Status()

    # Bootstrap sync
    comm_world.Barrier()

    time.sleep(0.1)
    file_path = glob("files/*/activities.csv")[0]
    flag = True
    index = 0
    old_value = 10
    if message_count_target == 0:
        while flag:
            df = pd.read_csv(file_path)
            if len(df[index:]) >= threshold:
                new_value = np.nanmean(df[index:].quality)
            if abs(new_value - old_value) <= convergence_param:
                data = np.array([1], dtype="i")
                req = comm_world.Isend(
                    data, dest=rank_index["data_manager"]
                )  # non blocking send
                req.Wait()
                flag = False
                break
            index += threshold
            old_value = new_value

    print(f"Convergence monitor stop @ rank: {rank}")
