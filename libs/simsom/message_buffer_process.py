"""
Message buffer process module.
"""

from mpi4py import MPI
from tqdm import tqdm
import datetime
import time


def message_buffer(
    comm_world: MPI.Intracomm,
    rank: int,
    max_iteration: int,
    sigterm: str = "STOP",
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

    log_msg = f"[{str(datetime.datetime.now())}] -> Running GLOBAL MESSAGE BUFFER process @ RANK {rank}..."
    print(log_msg, flush=True)
    time.sleep(3)

    update_queue = []

    pbar = tqdm(
        total=None if max_iteration <= 0 else max_iteration,
        desc="Running simulation",
        colour="green",
    )

    # Create status object once
    status = MPI.Status()

    while True:

        # Check if there is any message available (non-blocking)
        flag = comm_world.Iprobe(source=1, tag=MPI.ANY_TAG, status=status)

        if flag:

            # A message is available, so receive it
            message_object = comm_world.recv(
                source=status.Get_source(), tag=status.Get_tag()
            )

            # Check for termination signal
            if message_object == sigterm:
                comm_world.send(sigterm, dest=3)
                break

            # Add the received message to the update queue
            update_queue.append(message_object)
            pbar.update()  # Update progress bar by 1

            # If queue reaches 100 items, send it to the data manager
            if len(update_queue) == 100:
                comm_world.send(update_queue, dest=3)
                update_queue = []

        else:

            # If no message is available, you can sleep for a short time
            # time.sleep(0.1)  # Small delay to avoid busy waiting

            # If no message is available just force refresh the current bar
            pbar.refresh()

    # Ensure the progress bar reaches 100%
    pbar.close()

    log_msg = f"[{str(datetime.datetime.now())}] -> Closed GLOBAL MESSAGE BUFFER process @ RANK {rank}."
    print(log_msg, flush=True)
