"""
An agent receives inventory of messages (agent/user object) from the agent_pool_manager
and post/repost messages that will be shown to their followers
"""

from mpi4py import MPI
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


def run_agent(
    comm_world: MPI.Intracomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    print(f"* Agent process @rank: {rank} >> running...", flush=True)

    # Status of the processes
    status = MPI.Status()

    # Bootstrap sync
    comm_world.barrier()

    while True:

        # Receive package that contains (friend ids, messages) from agent_pool_manager
        # Wait for agent pack to process
        data = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

        # Check for termination
        if status.Get_tag() == 99:
            print(
                f"Agent@{rank} >> (1) sigterm detected, entering barrier... ",
                flush=True,
            )
            # flush_incoming_messages(comm_world, status)
            comm_world.barrier()
            break

        # # Check if the data is a termination signal and break the loop propagating the sigterm
        # if data == "sigterm":
        #     # print("- Agent process >> termination signal, stopping simulation...")
        #     comm_world.send(data, dest=rank_index["policy_filter"])
        #     # Flush pending incoming messages so we can exit cleanly
        #     while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
        #         _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
        #     comm_world.Barrier()
        #     break

        user = data
        activities, passivities = user.make_actions()

        # Repack the agent (updated feed) and activities (messages he produced)
        agent_pack_reply = ("agent_proc", (user, activities, passivities))

        req1 = comm_world.isend(agent_pack_reply, dest=rank_index["data_manager"])

        # Send the processed user to the policy process
        req2 = comm_world.isend(user, dest=rank_index["policy_filter"])

        safe_finalize_isends([req1, req2])

    print(f"* Agent process @rank: {rank} >> closed.", flush=True)
