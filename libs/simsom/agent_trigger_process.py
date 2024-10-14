"""
Agent activator process.
"""

import time
from mpi4py import MPI


def agent_trigger(comm_world: MPI.Intracomm, rank: int, sigterm: str = "STOP") -> None:
    """
    Agent activator process (processes 3 to M) simulates the activity of a user.
    In its current state it takes care of receiving a user_id from agent_process_manager,
    Generates a random number
    Returns: a tuple (User object, process rank).

    Args:
        comm (MPI.COMM_WORLD): communication context between processes
        rank (int): rank of the process

    """

    while True:
        # get user_id from agent_process_manager, check for termination and
        # send the action back to agent_process_manager
        user = comm_world.recv(source=1)
        if user == sigterm:
            break
        user.perform_action(
            str(user.user_id) + "_" + str(user.n_actions), user.is_shadow
        )
        # time.sleep(0.01)
        comm_world.send((user, rank), dest=1)
