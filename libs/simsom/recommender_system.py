from mpi4py import MPI
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
import random
from collections import Counter

def calculate_cosine_similarity(list_a: list, list_b: list) -> float:
    """
    Calculate the cosine similarity between two lists.
    """

    a_vals = Counter(list_a)
    b_vals = Counter(list_b)
    
    # convert to word-vectors
    words  = list(a_vals.keys() | b_vals.keys())
    a_vect = [a_vals.get(word, 0) for word in words]       
    b_vect = [b_vals.get(word, 0) for word in words]        

    # find cosine
    len_a  = sum(av*av for av in a_vect) ** 0.5             
    len_b  = sum(bv*bv for bv in b_vect) ** 0.5             
    dot    = sum(av*bv for av,bv in zip(a_vect, b_vect))   
    cosine = dot / (len_a * len_b) 
    return cosine

def run_recommender_system(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,  # If needed for future logic
    rank_index: dict,
):

    # Verbose: use flush=True to print messages
    # print("- RecSys process >> started", flush=True)

    # Status of the processes
    status = MPI.Status()

    global_inventory = []

    # Function to check for termination signal
    # (non-blocking)
    def check_for_sigterm():
        """Non-blocking check for termination signal"""
        if comm_world.Iprobe(source=rank_index["analyzer"]):
            _ = comm_world.recv(source=rank_index["analyzer"])
            return True
        return False

    def sort_based_topics(messages: list, agent) -> list:
        if len(messages) == 0:
            return messages
        user_topics = agent.user_topics

        # Calculate the cosine similarity between the user's topics and the messages
        similarities = [(calculate_cosine_similarity(user_topics, message.topics), message) for message in messages]

        # Sort messages by similarity score, descending
        ranked = sorted(similarities, key=lambda x: x[0], reverse=True)

        # Return just the sorted messages
        return [msg for _, msg in ranked]



    def build_feed(agent, in_messages, out_messages, in_perc=0.5, out_perc=0.5) -> list:
        """
        Build the newsfeed for the agent based on the incoming and outgoing messages.
        """
        # If there are no messages, return an empty list
        if not in_messages and not out_messages:
            return []
        # Sort the messages based on topics
        # in_messages = sort_based_topics(in_messages, agent)
        # out_messages = sort_based_topics(out_messages, agent)

        # Get percentages of messages to keep from in and out
        n_in = int(len(in_messages) * in_perc)
        n_out = int(len(out_messages) * out_perc)
        # Build the newsfeed and shuffle it
        new_feed = in_messages[:n_in] + out_messages[:n_out]
        new_feed = clean_feed(new_feed)
        new_feed = sort_based_topics(new_feed, agent)
        # Cut off the newsfeed if needed
        if len(new_feed) > agent.cut_off:
            new_feed = new_feed[: agent.cut_off]
        agent.newsfeed = new_feed
        return agent.newsfeed

    def clean_feed(newsfeed):
        """
        Clean the newsfeed for the agent removing duplicates
        """
        weight_dict = {}
        # Sort messages and drop duplicates (reshare)
        sorted_messages = sorted(newsfeed, key=lambda x: x.time, reverse=True)
        message_filter_dict = {}
        nan_parents = []
        # Iterate to check if there are duplicated reshare messages
        for message in sorted_messages:
            if message.reshared_original_id is None:
                nan_parents.append(message)
                # print(nan_parents)
            else:
                # check for duplicates and if they are present keep track of the weight (n of time they appear)
                if message.reshared_original_id not in message_filter_dict:
                    message_filter_dict[message.reshared_original_id] = message
                    weight_dict[message.reshared_original_id] = 1
                else:
                    weight_dict[message.reshared_original_id] += 1
        new_newsfeed = list(message_filter_dict.values()) + nan_parents

        # Sort list temporally and based on the weight
        new_newsfeed = sorted(new_newsfeed, key=lambda x: x.time, reverse=True)
        new_newsfeed = sorted(
            new_newsfeed,
            key=lambda x: (weight_dict.get(x.reshared_original_id, 0), x.time),
            reverse=True,
        )
        return new_newsfeed

    # Close the process cleanly
    def close_process():
        # print("- RecSys >> termination signal, stopping simulation...", flush=True)

        comm_world.send(("sigterm", 0), dest=rank_index["data_manager"])
        comm_world.send("sigterm", dest=rank_index["agent_pool_manager"])

        # Flush pending incoming messages
        while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
            _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
        comm_world.Barrier()

    # Bootstrap sync
    comm_world.Barrier()

    while True:

        # Check for termination signal (we need two of them because we risk
        # to miss the first one if we are busy processing data)
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
        users_with_time =[]
        passivities = []
        activities = []
        # Unpack the data and iterate over the contents
        for user, active_actions, passive_actions, current_time in data:
            # Get the message from inside and outside the network
            in_messages = []
            out_messages = []
            # Keep track of the messages using a global inventory
            global_inventory.extend(active_actions)
            for activity in global_inventory:
                if activity.uid in user.friends:
                    in_messages.append(activity)
                else:
                    out_messages.append(activity)
            # Build the newsfeed for the agent 
            user.newsfeed = build_feed(user, in_messages, out_messages)
            # Collect the user and the actions so we can send them to the agent pool manager and analyzer
            users_with_time.append((user, current_time)) #appending current_time as well 
            passivities.extend(passive_actions)
            activities.extend(active_actions)
        
        if len(global_inventory) > 2000:
            # Remove the oldest 1000 messages so we don't run out of memory
            global_inventory = global_inventory[-1000:] 
        
        # Clean newsfeeds of messages from suspended/terminated users
        suspended_ids = set(
            user.uid for user, _ in users_with_time if user.is_suspended or user.is_terminated
        )
        
        # Clean the feeds in-place and rebuild the cleaned 'users' list
        users = []
        for user, current_time in users_with_time:
            user.newsfeed = [msg for msg in user.newsfeed if msg.uid not in suspended_ids]
            users.append(user)  # Rebuild the cleaned version for analyzer

        # Check for termination signal (we need two of them because we risk
        # to miss the first one if we are busy processing data)
        if check_for_sigterm():
            close_process()
            break
        
        comm_world.send((users, activities, passivities), dest=rank_index["analyzer"])
        comm_world.send(users_with_time, dest=rank_index["agent_pool_manager"])
