"""
Contains common utility function for low level implmentation.
"""

from mpi4py import MPI
import time


def iprobe_with_timeout(
    comm_world,
    *,
    source=MPI.ANY_SOURCE,
    tag=MPI.ANY_TAG,
    timeout=3.0,
    check_interval=0.001,
    status=None,
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

    return False
