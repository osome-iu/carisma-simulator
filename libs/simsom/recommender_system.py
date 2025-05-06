from mpi4py import MPI
import time
import numpy as np
import random

def run_recommender_system(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,  # If needed for future logic
    rank_index: dict,
):

    # Verbose: use flush=True to print messages
    print("- RecSys process >> started", flush=True)

    # Status of the processes
    status = MPI.Status()

    global_inventory = []

    def check_for_sigterm():
        """Non-blocking check for termination signal"""
        if comm_world.Iprobe(source=rank_index["analyzer"]):
            _ = comm_world.recv(source=rank_index["analyzer"])
            return True
        return False


    def build_feed(agent, in_messages, out_messages, in_perc=1, out_perc=0) -> list:
        """
        Build the newsfeed for the agent based on the incoming and outgoing messages.
        """
        if not in_messages and not out_messages:
            return []
        random.shuffle(in_messages)
        random.shuffle(out_messages)
        n_in = int(len(in_messages) * in_perc)
        n_out = int(len(out_messages) * out_perc)
        new_feed = in_messages[:n_in] + out_messages[:n_out]
        random.shuffle(new_feed)
        if len(new_feed) > agent.cut_off:
            new_feed = new_feed[: agent.cut_off]
        agent.newsfeed = new_feed
        return agent.newsfeed

    # Close the process cleanly
    def close_process():
        print("- RecSys >> termination signal, stopping simulation...", flush=True)

        comm_world.send(("sigterm", 0), dest=rank_index["data_manager"])
        comm_world.send("sigterm", dest=rank_index["agent_pool_manager"])

        # Flush pending incoming messages
        while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
            _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
        comm_world.Barrier()

    # Bootstrap sync
    comm_world.Barrier()

    while True:

        if check_for_sigterm():

            close_process()
            break
        
        data = comm_world.recv(source=rank_index["agent_pool_manager"], status=status)

        # Wait untile we receive data from the agent pool manager (agent pool manager may have not 
        # enough users ready to pick them up so it will send empty list)
        comm_world.send(("ping_recsys", 0), dest=rank_index["data_manager"])
        data = comm_world.recv(source=rank_index["data_manager"], status=status)
        # print("- RecSys >> data received.", flush=True)
        # print(data)
        users = []
        passivities = []
        activities = []
        # Check if the data is empty, so first iteration and just send the user
        for user , active_actions, passive_actions in data:
            in_messages = []
            out_messages = []
            global_inventory.extend(active_actions)
            for activity in global_inventory:
                if activity.uid in user.friends:
                    in_messages.append(activity)
                else:
                    out_messages.append(activity)
            # Build the newsfeed for the agent 
            user.newsfeed = build_feed(user, in_messages, out_messages)
            users.append(user)
            passivities.extend(passive_actions)
            activities.extend(active_actions)
        
        if len(global_inventory) > 2000:
            # Remove the oldest 1000 messages
            global_inventory = global_inventory[-1000:] 
            
        if check_for_sigterm():
            close_process()
            break
        
        comm_world.send((activities, passivities), dest=rank_index["analyzer"])
        comm_world.send(users, dest=rank_index["agent_pool_manager"])
        # print("- RecSys >> data sent.", flush=True)
        
    print("- RecSys >> finished", flush=True)