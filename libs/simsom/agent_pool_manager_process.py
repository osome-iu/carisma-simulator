"""
An agent pool manager handle the pool of parallel running agent processes.
Main task is to dispatch User/Agent objects to agent processes.
"""

import random as rnd
from mpi4py import MPI
from mpi_utils import iprobe_with_timeout


def run_agent_pool_manager(
    comm_world: MPI.Intracomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    print("* Agent pool manager >> running...", flush=True)

    # Status of the processes
    status = MPI.Status()

    # Ranks of all available agent handler
    agent_handlers_ranks = list(range(rank_index["agent_handler"], size))
    # print("- Agent Pool Manager >> agent process ranks", agent_handlers_ranks, flush=True)

    # Process status
    alive = True

    # Process isends
    isends = []

    # Bootstrap sync
    comm_world.barrier()

    while True:

        if alive:
            # Request data from recommender system process
            req1 = comm_world.isend(
                "agents_needed",
                dest=rank_index["recommender_system"],
            )
            isends.append(req1)

        if iprobe_with_timeout(
            comm_world,
            source=MPI.ANY_SOURCE,
            tag=MPI.ANY_TAG,
            status=status,
        ):

            # Receive incoming data (from any process is sending)
            data = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

            # Check for termination
            if status.Get_tag() == 99:
                print("* AgntPoolMngr >> stop signal detected", flush=True)
                alive = False

            # Wait for pending isends
            MPI.Request.waitall(isends)
            isends.clear()

            if alive:

                for user in data:
                    handler_rank = rnd.choice(agent_handlers_ranks)
                    req = comm_world.isend(user, dest=handler_rank)
                    isends.append(req)

        else:
            print("* AgntPoolMngr >> waiting isends...", flush=True)
            MPI.Request.waitall(isends)
            print("* AgntPoolMngr >> entering barrier...", flush=True)
            comm_world.barrier()
            break

    print("* Agent pool manager >> closed.", flush=True)
