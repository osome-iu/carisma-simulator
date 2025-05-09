"""
Reponsible for monitoring the convergence of the simulation.
Send termination signal to all processes when the simulation has converged.
"""

import time
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
    last_idx = df[df['user_id'] == last_user].last_valid_index()
    df = df.loc[:last_idx]
    df.to_csv(
        file_path_passivity,
        lineterminator="\n",
        index=False,
        encoding="utf-8",
    )

def update_quality(current_quality, overall_avg_quality) -> None:
    """
    Update quality using exponential moving average to ensure stable state at convergence
    Forget the past slowly, i.e., new_quality = 0.8 * avg_quality(at t-1) + 0.2 * current_quality
    """

    new_quality = (rho * current_quality + (1 - rho) * overall_avg_quality)
    quality_diff = (
        abs(new_quality - current_quality) / current_quality if current_quality > 0 else 0
    )
    return quality_diff, new_quality

def run_analyzer(
    comm_world: MPI.Intercomm,
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
    ema_quality: bool,
    ema_quality_convergence: float,
    # Number of users to be used for the simulation
    n_users: int,
    # Params for printing stuff during the execution
    verbose: bool,
    print_interval: int,
    # Params for saving activities on disk
    save_active_interactions: bool=True,
    save_passive_interactions: bool=True,  

):
    """
    Function that takes care of calculating the convergence condition and stop execution

    Args:
        comm (MPI.COMM_WORLD): communication context between processes
        sliding_window_convergence (int): sliding window size for quality analysis
        FILE_PATH (str): path to the file where the activities are saved
    """

    # Verbose: use flush=True to print messages
    # print("- Analyzer >> started", flush=True)

    status = MPI.Status()

    n_data = 0                  # keep track of the number of messages
    intermediate_n_user = 0     # keep track of the number of users
    interval_quality = 0        # keep track of the quality of the messages for interval printing
    quality_sum = 0             # keep track of the quality of the messages
    count = 0                   # keep track of the number of messages for verbose debug
    current_quality_list = []   # list of qualities for the sliding window
    previous_quality = None     # quality of the previous window
    feeds = {}                  # dictionary of feeds for the users, this is used to calculate diversity, quality, etc.
    users = []                  # list of users for the ema quality
    current_quality = 1         # value to calculate the current quality each N iterations (or after T time)
    

    # Files for writing the activities
    csv_out_act = None
    out_act = None
    
    if max_interactions_method and sliding_window_method and ema_quality:
        # Since we need to choose one of the two methods, we will use the max_interactions_method
        sliding_window_method = False
        ema_quality = False
    
    if max_interactions_method:
        exec_name = 'Max iterations'
    elif sliding_window_method:
        exec_name = 'Sliding windows'
    else:
        exec_name = 'Exponential moving average'

    
    print(f"Execution with {exec_name}")
    
    # Initialize files
    simtools.init_files(folder_path, file_path_activity, file_path_passivity)

    # Bootstrap sync
    comm_world.Barrier()

    # Function to terminate the process and print information
    def clean_termination() -> None:
        """Clean termination of the process"""
        # print("- Analyzer >> GOAL REACHED, TERMINATING SIMULATION...", flush=True)
        comm_world.send("sigterm", dest=rank_index["recommender_system"])
        # print("- Analyzer >> sent termination signal to recommender system", flush=True)
        # Flush pending incoming messages
        while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
            _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
        comm_world.Barrier()
        # print("- Analyzer >> flushed pending messages", flush=True)

    while True:

        # Get data from policy filter
        data = comm_world.recv(source=rank_index["recommender_system"], status=status)
        # Unpack the data
        user, activities, passivities = data
        # Count the number of messages
        n_data += len(activities)
        intermediate_n_user += 1

        # Write the data to the files
        # Write the active interactions (post/repost)
        if save_active_interactions:
            out_act = open(file_path_activity, "a", newline="", encoding="utf-8")
            csv_out_act = csv.writer(out_act)
        try:
            for m in activities:
                quality_sum += m.quality
                interval_quality += m.quality
                count += 1
                if csv_out_act:
                    csv_out_act.writerow(m.write_action())
        finally:
            if out_act:
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
            if intermediate_n_user % print_interval == 0:
                print(f"Intermediate stats after {intermediate_n_user} users: interval quality --> {round(interval_quality / count, 2)}", flush=True)
                count = 0
                interval_quality = 0

        # Based on the method for convergence check if we should stop
        if max_interactions_method:
            # Stop and terminate the process
            if n_data >= max_iteration_target:
                clean_termination()
                # Resize the output file to the number of messages
                resize_output(max_iteration_target)
                print("Average quality:", round(quality_sum / n_data, 2), flush=True)
                break

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
                    if abs(current_quality - previous_quality) <= sliding_window_threshold:
                        clean_termination()
                        # Resize the output file to the number of messages
                        resize_output(n_data)
                        print("Threshold reached:", abs(current_quality - previous_quality), flush=True)
                        print("Average quality:", round(quality_sum / n_data, 2), flush=True)
                        break
                # Update the previous quality
                previous_quality = current_quality
                current_quality_list = []
        # Use the convergence with exponential moving average
        elif ema_quality:
            users.append(user)
            feeds[user.uid] = user.newsfeed
            if len(users) == n_users:
                users = []
                quality_diff, new_quality = update_quality(current_quality=current_quality, overall_avg_quality=quality_sum / n_data)
                current_quality = new_quality
                if quality_diff <= ema_quality_convergence:
                    clean_termination()
                    # Resize the output file to the number of messages
                    resize_output(max_iteration_target)
                    print("Average quality:", round(quality_sum / n_data, 2), flush=True)
                    break
                else:
                    print(f"Quality diff after {n_data} messages: {quality_diff}")