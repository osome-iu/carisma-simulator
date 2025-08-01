"""
An agent pool manager handle the pool of parallel running agent processes.
Main task is to dispatch User/Agent objects to agent processes.
"""

import random as rnd
from mpi4py import MPI
from mpi_utils import iprobe_with_timeout, handle_crash


def run_agent_pool_manager(
    comm_world: MPI.Intracomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    print("* Agent pool manager >> running...", flush=True)

    # Status of the processes
    status = MPI.Status()

    # Ranks of all available workers
    agent_handlers_ranks = list(range(rank_index["agent_handler"], size))
    print(
        "* Agent Pool Manager >> available workers:",
        agent_handlers_ranks,
        flush=True,
    )

    # Process status
    alive = True

    # Process isends
    isends = []

    # Bootstrap sync
    comm_world.barrier()

    while True:

        if alive:

            # Request data from recommender system process
            isends.append(
                comm_world.isend(
                    ("agntPoolMngr", "dataReq"),
                    dest=rank_index["recommender_system"],
                )
            )

        if iprobe_with_timeout(
            comm_world,
            source=MPI.ANY_SOURCE,
            tag=MPI.ANY_TAG,
            status=status,
            pname="AgentPoolMngr",
        ):

            # Receive incoming data (from any process is sending)
            sender, payload = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

            # Check if termination signal has been sent
            if alive and payload == "STOP":
                print("* AgentPoolMngr >> stop signal detected...", flush=True)
                MPI.Request.waitall(isends)
                isends.clear()
                alive = False

            if alive:

                # Wait for pending isends
                if len(isends) > 100 or not alive:
                    MPI.Request.waitall(isends)
                    isends.clear()

                for user in payload:
                    handler_rank = rnd.choice(agent_handlers_ranks)
                    isends.append(
                        comm_world.isend(
                            ("agntPoolMngr", user),
                            dest=handler_rank,
                        )
                    )

        else:

            print(
                f"* AgentPoolMngr >> closing with {len(isends)} isends...", flush=True
            )

            if alive:

                MPI.Request.waitall(isends)

                handle_crash(
                    comm_world=comm_world,
                    status=status,
                    srank=rank,
                    srole="agent_pool_manager",
                    pname="AgentPoolMngr",
                )

            print("* AgentPoolMngr >> entering barrier...", flush=True)
            comm_world.barrier()
            break

    print("* AgentPoolMngr >> closed.", flush=True)
