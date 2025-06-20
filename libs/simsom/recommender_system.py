from mpi4py import MPI
from collections import Counter
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


# def flush_incoming_messages(comm, status):
#     while comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
#         _ = comm.recv(source=status.Get_source(), tag=status.Get_tag(), status=status)


def calculate_cosine_similarity(list_a: list, list_b: list) -> float:
    """
    Calculate the cosine similarity between two lists.
    """

    a_vals = Counter(list_a)
    b_vals = Counter(list_b)

    # convert to word-vectors
    words = list(a_vals.keys() | b_vals.keys())
    a_vect = [a_vals.get(word, 0) for word in words]
    b_vect = [b_vals.get(word, 0) for word in words]

    # find cosine
    len_a = sum(av * av for av in a_vect) ** 0.5
    len_b = sum(bv * bv for bv in b_vect) ** 0.5
    dot = sum(av * bv for av, bv in zip(a_vect, b_vect))
    cosine = dot / (len_a * len_b)
    return cosine


def run_recommender_system(
    comm_world: MPI.Intracomm,
    rank: int,
    size: int,  # If needed for future logic
    rank_index: dict,
):

    print("* RecSys process >> running...", flush=True)

    # Status of the processes
    status = MPI.Status()

    global_inventory = []

    # # Function to check for termination signal
    # # (non-blocking)
    # def check_for_sigterm():
    #     """Non-blocking check for termination signal"""
    #     if comm_world.Iprobe(source=rank_index["analyzer"]):
    #         _ = comm_world.recv(source=rank_index["analyzer"])
    #         return True
    #     return False

    def sort_based_topics(messages: list, agent) -> list:
        if len(messages) == 0:
            return messages
        user_topics = agent.user_topics

        # Calculate the cosine similarity between the user's topics and the messages
        similarities = [
            (calculate_cosine_similarity(user_topics, message.topics), message)
            for message in messages
        ]

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

    # # Close the process cleanly
    # def close_process():
    #     # print("- RecSys >> termination signal, stopping simulation...", flush=True)

    #     comm_world.send(("sigterm", 0), dest=rank_index["data_manager"])
    #     comm_world.send("sigterm", dest=rank_index["agent_pool_manager"])

    #     # Flush pending incoming messages
    #     while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
    #         _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
    #     comm_world.barrier()

    # Bootstrap sync
    comm_world.barrier()

    while True:

        # # Check for termination signal (we need two of them because we risk
        # # to miss the first one if we are busy processing data)
        # if check_for_sigterm():
        #     close_process()
        #     break

        # Wait for agent pool manager requesting data
        data = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

        # Check for termination
        if status.Get_tag() == 99:
            print("recsys >> (1) sigterm detected, entering barrier... ", flush=True)
            # flush_incoming_messages(comm_world, status)
            comm_world.barrier()
            break

        # Wait until we receive data from the agent pool manager (agent pool manager may have not
        # enough users ready to pick them up so it will send empty list)

        # Requesting data to data manager (non blocking)
        req1 = comm_world.isend(("recsys_proc", None), dest=rank_index["data_manager"])

        # Wait for data from data manager
        data = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

        # Check for termination
        if status.Get_tag() == 99:
            print("recsys >> (2) sigterm detected, entering barrier... ", flush=True)
            req1.cancel()
            # flush_incoming_messages(comm_world, status)
            comm_world.barrier()
            break

        users = []
        passivities = []
        activities = []
        # Unpack the data and iterate over the contents
        for user, active_actions, passive_actions in data:
            # print(
            #     f"Received from DataMngr {len(active_actions)} active actions.",
            #     flush=True,
            # )
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
            users.append(user)
            passivities.extend(passive_actions)
            activities.extend(active_actions)

        if len(global_inventory) > 2000:
            # Remove the oldest 1000 messages so we don't run out of memory
            global_inventory = global_inventory[-1000:]

        # # Check for termination signal (we need two of them because we risk
        # # to miss the first one if we are busy processing data)
        # if check_for_sigterm():
        #     close_process()
        #     break

        # print(f"Sending {len(activities)} activities to analyzer", flush=True)
        # print(f"Sending {len(users)} users to analyzer", flush=True)

        analyzer_pack = (users, activities, passivities)

        req2 = comm_world.isend(analyzer_pack, dest=rank_index["analyzer"])
        req3 = comm_world.isend(users, dest=rank_index["agent_pool_manager"])

        safe_finalize_isends([req1, req2, req3])

    print("* RecSys >> closed.", flush=True)
