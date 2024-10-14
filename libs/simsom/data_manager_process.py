"""
Data manager process module.
"""

import csv
from mpi4py import MPI
import datetime
import time


def data_manager(comm: MPI.Intracomm, rank: int, file_path: str, sigterm: str) -> None:
    """
    Data_manager (process 3) receives a structure
    containing user activities from update_queue_manager,
    assigns a time T to it via the clock function, and writes the data to memory.

    Args:
        comm (MPI.COMM_WORLD): communication context between processes
    """

    # NOTE: Open file stream out of the while and close when sigterm is used

    log_msg = f"[{str(datetime.datetime.now())}] -> Running DATA MANAGER process @ RANK {rank}..."
    print(log_msg, flush=True)
    time.sleep(3)

    while True:
        update_queue = comm.recv(source=2)
        if update_queue == sigterm:
            break
        # add in the tuple the clock time and write the data to the file
        with open(file_path, "a", newline="", encoding="utf-8") as out:
            csv_out = csv.writer(out)
            for row in update_queue:
                csv_out.writerow(row.write_to_disk())

    log_msg = f"[{str(datetime.datetime.now())}] -> Closed DATA MANAGER process @ RANK {rank}."
    print(log_msg, flush=True)
