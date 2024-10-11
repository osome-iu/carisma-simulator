"""
Data manager process module.
"""

import csv
from mpi4py import MPI


def data_manager(comm: MPI.Intracomm, file_path: str, sigterm: str) -> None:
    """
    Data_manager (process 3) receives a structure
    containing user activities from update_queue_manager,
    assigns a time T to it via the clock function, and writes the data to memory.

    Args:
        comm (MPI.COMM_WORLD): communication context between processes
    """

    while True:
        update_queue = comm.recv(source=2)
        if update_queue == sigterm:
            break
        # add in the tuple the clock time and write the data to the file
        with open(file_path, "a", newline="", encoding="utf-8") as out:
            csv_out = csv.writer(out)
            for row in update_queue:
                csv_out.writerow(row.write_to_disk())
