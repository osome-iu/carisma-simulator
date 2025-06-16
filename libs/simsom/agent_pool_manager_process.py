"""
An agent pool manager handle the pool of parallel running agent processes.
Main task is to dispatch User/Agent objects to agent processes.
"""

from mpi4py import MPI
import random as rnd
import time


def safe_finalize_isends(requests, soft_checks=3, hard_timeout=0.1):
    pending = requests.copy()
    for _ in range(soft_checks):
        if MPI.Request.Testall(pending):
            return
    # fallback with sleep
    start = time.time()
    while time.time() - start < hard_timeout and pending:
        pending = [req for req in pending if not req.Test()]


def flush_incoming_messages(comm, status):
    while comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
        _ = comm.recv(source=status.Get_source(), tag=status.Get_tag(), status=status)


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

    # Bootstrap sync
    comm_world.barrier()

    while True:

        # Request data from recommender system process
        req1 = comm_world.isend("agents_needed", dest=rank_index["recommender_system"])

        # Wait for data from anyone
        data = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

        # Check for termination
        if status.Get_tag() == 99:
            print("AgntPoolMngr >> sigterm detected (1)", flush=True)
            req1.cancel()
            flush_incoming_messages(comm_world, status)
            comm_world.barrier()
            break

        # # Check for termination
        # if data == "sigterm":
        #     # Send termination signal to all agent handlers and print message
        #     # print("- Agent Pool Manager >> termination signal", flush=True)
        #     for i in range(rank_index["agent_handler"], size):
        #         comm_world.send("sigterm", dest=i)

        #     # Flush pending incoming messages so we can exit cleanly
        #     while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
        #         _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
        #     comm_world.Barrier()
        #     break

        # If is not termination signal, dispatch data to agent handlers
        dispatch_requests = []

        for user in data:
            handler_rank = rnd.choice(agent_handlers_ranks)
            req = comm_world.isend(user, dest=handler_rank)
            dispatch_requests.append(req)

        # MPI.Request.waitall(dispatch_requests)
        safe_finalize_isends(dispatch_requests)

    print("* Agent pool manager >> closed.", flush=True)
