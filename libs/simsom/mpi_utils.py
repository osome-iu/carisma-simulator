"""
Contains common utility function for low level implmentation.
"""

from mpi4py import MPI

# import random
import time
from datetime import datetime


def gettimestamp():
    now = datetime.now()
    # return now.strftime("%Y-%m-%d %H:%M:%S")
    return now.strftime("%Y-%m-%d %H:%M:%S.%f")


def iprobe_with_timeout(
    comm_world: MPI.Intracomm,
    source=MPI.ANY_SOURCE,
    tag=MPI.ANY_TAG,
    status=None,
    timeout=10.0,
    check_interval=0.001,
    pname="Proc",
):
    """
    Non-blocking probe with timeout.

    Parameters:
    - comm_world (MPI.Comm): The communicator (e.g., MPI.COMM_WORLD)
    - source (int): The source rank or MPI.ANY_SOURCE (default)
    - tag (int): The message tag or MPI.ANY_TAG (default)
    - timeout (float): Maximum time to wait in seconds
    - check_interval (float): Sleep duration between checks (in seconds)
    - status (MPI.Status, optional): An MPI.Status object to fill on success

    Returns:
    - bool: True if a message was probed within the timeout window, False otherwise
    """
    start = time.perf_counter()

    while time.perf_counter() - start < timeout:
        if comm_world.iprobe(source=source, tag=tag, status=status):
            return True
        time.sleep(check_interval)

    print(f"[{gettimestamp()}] {pname} >> timeout", flush=True)

    return False


def clean_termination(
    comm_world,
    sender_rank: int,
    sender_role: str,
    log_name: str,
    message: str,
):
    """
    Broadcast a termination signal from the sender process to all others.

    Parameters:
    - comm: MPI communicator (e.g., MPI.COMM_WORLD)
    - sender_rank: int, rank of the process sending the termination signal
    - sender_role: str, identifier of the role of the sender (for logging)
    """
    print(f"[{gettimestamp()}] {log_name} > {message}", flush=True)

    proc_ranks = list(range(comm_world.Get_size()))

    # random.shuffle(
    #     proc_ranks
    # )  # Shuffle to simulate non-deterministic signal order (optional)

    isends = []

    for rank in proc_ranks:
        if rank != sender_rank:
            isends.append(comm_world.isend((sender_role, "STOP"), dest=rank))
            print(
                f"[{gettimestamp()}] {log_name} >> sent termination signal to: {rank}",
                flush=True,
            )

    MPI.Request.waitall(isends)

    print(f"[{gettimestamp()}] {log_name} >> sent all termination signals!", flush=True)


def handle_crash(comm_world, status, srank: int, srole: str, pname: str):

    # Notify all other processes
    clean_termination(
        comm_world=comm_world,
        sender_rank=srank,
        sender_role=srole,
        log_name=pname,
        message="crashing...",
    )

    # Switch to consumer mode
    while iprobe_with_timeout(
        comm_world=comm_world,
        status=status,
        pname=pname,
        timeout=5,
    ):

        _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
