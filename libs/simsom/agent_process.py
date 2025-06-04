"""
An agent receives inventory of messages (agent/user object) from the agent_pool_manager
and post/repost messages that will be shown to their followers
"""

import numpy as np
from mpi4py import MPI
import time


def run_agent(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    # Verbose: use flush=True to print messages
    # print(f"- Agent process @{rank} >> started", flush=True)

    # Status of the processes
    status = MPI.Status()

    # Bootstrap sync
    comm_world.Barrier()

    while True:

        # Receive package that contains (friend ids, messages) from agent_pool_manager
        # Wait for agent pack to process
        data = comm_world.recv(
            source=rank_index["agent_pool_manager"],
            status=status,
        )

        # Check if the data is a termination signal and break the loop propagating the sigterm
        if data == "sigterm":
            # print("- Agent process >> termination signal, stopping simulation...")
            comm_world.send(data, dest=rank_index["policy_filter"])
            # Flush pending incoming messages so we can exit cleanly
            while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
                _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
            comm_world.Barrier()
            break
        user, current_time = data
        
        new_msgs, passive_actions = user.make_actions()
        
        # Repack the agent (updated feed) and actions (messages he produced)
        agent_pack_reply = (user, new_msgs, passive_actions)


        comm_world.send(("ping_agent_pool_manager", agent_pack_reply), dest=rank_index["data_manager"])
        comm_world.send(data, dest=rank_index["policy_filter"])
