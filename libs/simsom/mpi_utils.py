"""
Contains common utility function for low level implmentation.
"""

from mpi4py import MPI
import time


def isend_green_light(requests, soft_checks=3, hard_timeout=0.1):
    pending = requests.copy()

    # Fast loop for quick completion (non-blocking)
    for _ in range(soft_checks):
        if MPI.Request.Testall(pending):
            return True
        time.sleep(0.001)  # Tiny delay to avoid CPU burn

    # Fallback: slow loop with timeout
    start = time.time()
    while time.time() - start < hard_timeout and pending:
        pending = [req for req in pending if not req.Test()]
        time.sleep(0.001)

    return len(pending) == 0


def iprobe_with_timeout(
    comm_world,
    source=MPI.ANY_SOURCE,
    tag=MPI.ANY_TAG,
    timeout=1.0,
    check_interval=0.001,
    status=None,
):
    """
    Non-blocking probe with timeout.

    Parameters:
    - comm_world: The communicator (e.g., MPI.COMM_WORLD)
    - source: The source rank or MPI.ANY_SOURCE
    - tag: The message tag or MPI.ANY_TAG
    - timeout: Maximum time to wait in seconds
    - check_interval: Sleep duration between checks (in seconds)

    Returns:
    - True if a message was probed within the timeout window
    - False otherwise
    """
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if comm_world.iprobe(source=source, tag=tag, status=status):
            return True
        time.sleep(check_interval)  # lightweight pause to avoid CPU burn
    return False


def iprobe_with_timeout_v2(
    comm,
    timeout=1.0,
    source=MPI.ANY_SOURCE,
    tag=MPI.ANY_TAG,
    status=None,
):
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if comm.iprobe(source=source, tag=tag, status=status):
            return True
        time.sleep(0.001)  # Prevents CPU burn without slowing real traffic
    return False
