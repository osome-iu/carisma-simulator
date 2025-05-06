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
import simtools
import time
import pandas as pd

# Path files
time_now = int(time.time())
folder_path = f"files/{time_now}"
file_path_activity = folder_path + "/activities.csv"
file_path_passivity = folder_path + "/passivities.csv"



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



def run_analyzer(
    comm_world: MPI.Intercomm,
    rank: int,
    rank_index: dict,
    sliding_window_convergence: int,
    message_count_target: int,
    threshold_convergence: float,
    verbose: bool,
    print_interval: int,
    save_passive_interactions: bool=True
    
):
    """
    Function that takes care of calculating the convergence condition and stop execution

    Args:
        comm (MPI.COMM_WORLD): communication context between processes
        sliding_window_convergence (int): sliding window size for quality analysis
        FILE_PATH (str): path to the file where the activities are saved
    """

    # Verbose: use flush=True to print messages
    print("- Analyzer >> started", flush=True)

    status = MPI.Status()

    n_data = 0
    
    # Initialize files
    simtools.init_files(folder_path, file_path_activity, file_path_passivity)

    # Bootstrap sync
    comm_world.Barrier()

    def clean_termination():
        print("- Analyzer >> GOAL REACHED, TERMINATING SIMULATION...", flush=True)
        comm_world.send("sigterm", dest=rank_index["recommender_system"])
        print("- Analyzer >> sent termination signal to recommender system", flush=True)
        # Flush pending incoming messages
        while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
            _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
        comm_world.Barrier()
        print("- Analyzer >> flushed pending messages", flush=True)

    while True:

        # Get data from policy filter
        data = comm_world.recv(source=rank_index["recommender_system"], status=status)
        activities, passivities = data
        with open(
            file_path_activity, "a", newline="", encoding="utf-8"
        ) as out_act:
            csv_out_act = csv.writer(out_act)
            for m in activities:
                csv_out_act.writerow(m.write_action())
        # Write the passive interactions (view)
        if save_passive_interactions:
            with open(
                file_path_passivity, "a", newline="", encoding="utf-8"
            ) as out_pas:   
                csv_out_pas = csv.writer(out_pas)
                for a in passivities:
                    csv_out_pas.writerow(a.write_action())
        if message_count_target > 0:
            n_data += len(activities)

            if n_data >= message_count_target:
                clean_termination()
                resize_output(message_count_target)
                break
        else:
            # Use the convergence with sliding window or based on overleall messages
            continue
    print("- Analyzer >> simulation terminated", flush=True)
