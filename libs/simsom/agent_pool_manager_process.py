"""
An agent pool manager handle the pool of parallel running agent processes.
Main task is to dispatch User/Agent objects to agent processes.
"""

from mpi4py import MPI
import random as rnd
import time


def run_agent_pool_manager(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    # Verbose: use flush=True to print messages
    print("- Agent pool manager >> started", flush=True)

    # Status of the processes
    status = MPI.Status()

    # Ranks of all available agent handler
    agent_handlers_ranks = list(range(rank_index["agent_handler"], size))
    print(
        "- Agent Pool Manager >> agent process ranks", agent_handlers_ranks, flush=True
    )

    # Bootstrap sync
    comm_world.Barrier()

    while True:
        
        # Get data from recommender system process
        comm_world.send(
            "ping_agent_pool_manager",
            dest=rank_index["recommender_system"],
        )

        # Wait for data from recommender system process
        data = comm_world.recv(
            source=rank_index["recommender_system"],
            status=status,
        )

        # Check for termination
        if data == "sigterm":
            # Send termination signal to all agent handlers and print message
            print("- Agent Pool Manager >> termination signal", flush=True)
            for i in range(rank_index["agent_handler"], size):
                comm_world.send("sigterm", dest=i)

            # Flush pending incoming messages so we can exit cleanly
            while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
                _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
            comm_world.Barrier()
            break

        # If is not termination signal, dispatch data to agent handlers
        dispatch_requests = []

        for user in data:
            handler_rank = rnd.choice(agent_handlers_ranks)
            req = comm_world.isend(user, dest=handler_rank)
            dispatch_requests.append(req)

        MPI.Request.waitall(dispatch_requests)
