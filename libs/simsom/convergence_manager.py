"""
Process that handles convergence if the maximum number of iterations is not specified
"""

import pandas as pd
import numpy as np
from mpi4py import MPI


def convergence(
    comm: MPI.COMM_WORLD,
    threshold: int,
    FILE_PATH: str,
    stop_iteration: int,
    convergence_param: float,
):
    """
    Function that takes care of calculating the convergence condition and stop execution

    Args:
        comm (MPI.COMM_WORLD): communication context between processes
        threshold (int): sliding window size for quality analysis
        FILE_PATH (str): path to the file where the activities are saved
    """
    flag = True
    index = 0
    old_value = 10
    if stop_iteration == 0:
        while flag:
            df = pd.read_csv(FILE_PATH)
            if len(df[index:]) >= threshold:
                new_value = np.nanmean(df[index:].quality)
                if abs(new_value - old_value) <= convergence_param:
                    data = np.array([1], dtype="i")
                    req = comm.Isend(data, dest=1)  # non blocking send
                    req.Wait()
                    flag = False
                    break
                index += threshold
                old_value = new_value
