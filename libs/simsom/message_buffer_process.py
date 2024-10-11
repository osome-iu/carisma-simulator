"""
Message buffer process module.
"""

import sys
from tqdm import tqdm
from mpi4py import MPI


def message_buffer(
    comm: MPI.Intracomm, max_iteration: int, sigterm: str = "STOP"
) -> None:
    """
    Update_queue_manager (process 2) is responsible for sending
    chunks of variable size to the data_manager.
    It receives data from agent_process_manager and stores it in a data structure,
    when the structure reaches a specific size, it's sent to data_manager and flushed.

    Args:
        comm (MPI.COMM_WORLD):  communication context between processes
        max_iteration (int): max number of iteration that the simulation does before it stops
    """

    update_queue = []
    if max_iteration <= 0:
        max_iteration = None

    pbar = tqdm(
        total=max_iteration,
        file=sys.stdout,
        desc="Running",
        colour="green",
    )

    while True:
        message_object = comm.recv(source=1)
        if message_object == sigterm:
            comm.send(sigterm, dest=3)
            break
        update_queue.append(message_object)
        pbar.update()
        # TODO: chose a chunk size
        if len(update_queue) == 100:
            comm.send(update_queue, dest=3)
            update_queue = []
    pbar.close()
