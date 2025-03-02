"""
An agent pool manager handle the pool of parallel running agent processes.
Main task is to dispatch User/Agent objects to agent processes.
"""

from mpi4py import MPI
import random as rnd


def run_agent_pool_manager(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    # Ranks of all available agent handler
    agent_handlers_ranks = list(range(rank_index["agent_handler"], size))

    # Status of the processes
    status = MPI.Status()

    # Bootstrap sync
    comm_world.Barrier()

    while True:

        # Get data from policy filter
        user_packs_batch = comm_world.recv(
            source=rank_index["policy_filter"],
            status=status,
        )

        # Check for termination
        if user_packs_batch == "sigterm":
            break

        dispatch_requests = []

        # Dispatch all the agent packs
        while user_packs_batch:

            # Pick agent pack from the batch
            user_pack = user_packs_batch.pop()

            # Pick handler at random with replacement
            handler_rank = rnd.choice(agent_handlers_ranks)
            # NOTE: The same handler could be issued with more than
            # one agent pack at a time that will be processed when ready.

            # Non-blocking dispatch
            req = comm_world.isend(
                user_pack,
                dest=handler_rank,
            )

            dispatch_requests.append(req)

        # Wait for all agent packs dispatched
        MPI.Request.waitall(dispatch_requests)

    # Handlers shutdown
    for i in range(rank_index["agent_handler"], size):
        comm_world.send("sigterm", dest=i)
