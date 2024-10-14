"""
Helper and utilities module.
Put here function that can be used by more than one process.
"""

import os
import csv
from user import User
import random
import igraph as ig


def init_network(net_size: int, p=0.5, k_out=3) -> list:
    """
    Create a network using a directed variant of the random-walk growth model
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.67.056104
    Inputs:
        - net_size (int): number of nodes in the desired network
        - k_out (int): average no. friends for each new node
        - p (float): probability for a new node to follow friends of a
        friend (models network clustering)
    """
    if net_size <= k_out + 1:  # if super small just return a clique
        return ig.Graph.Full(net_size, directed=True)

    graph = ig.Graph.Full(k_out, directed=True)
    for n in range(k_out, net_size):
        target = random.choice(graph.vs)
        friends = [target]
        n_random_friends = 0
        for _ in range(k_out - 1):
            if random.random() < p:
                n_random_friends += 1

        friends += random.sample(
            graph.successors(target), n_random_friends
        )  # return a list of vertex id(int)
        friends += random.sample(range(graph.vcount()), k_out - 1 - n_random_friends)

        graph.add_vertex(n)

        edges = [(n, f) for f in friends]

        graph.add_edges(edges)

    graph.vs["name"] = [f"u{v.index}" for v in graph.vs]

    users = dict()
    for node in graph.vs:
        # remember link direction is following
        friends = graph.successors(node.index)  # this returns the index of the vertex
        followers = graph.predecessors(node.index)
        user_i = User(
            user_index=node.index,
            user_id=graph.vs[node.index]["name"],
            user_class="user",
            friends=friends,
            followers=followers,
        )
        users[node.index] = user_i
    return users


def file_manager(folder_path, file_path) -> None:
    """
    Function to manage the creation and removal of the file that store the activities.
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if os.path.isfile(file_path):
        os.remove(file_path)
    with open(file_path, "w", newline="", encoding="utf-8") as out:
        csv_out = csv.writer(out)
        if os.stat(file_path).st_size == 0:
            csv_out.writerow(
                [
                    "user_id",
                    "message_id",
                    "quality",
                    "appeal",
                    "reshared_id",
                    "reshared_user_id",
                    "reshared_original_id",
                    "clock_time",
                ]
            )


def get_scores_below_threshold(target: User, threshold: float) -> bool:
    """example of possible policy implementation.
    This function is a sample of possible shadow-ban or ban of users
    Open the activity file to get activities of specific user and return True if the quality of the
    activities are below a specific threshold more than 2 times.

    Args:
        user (user): specific user to be checked
        threshold (float): value

    Returns:
        bool: True if user have more thas 2 activities below the threshold
    """
    return (
        sum(1 for message in target.shared_messages if message.quality < threshold) >= 3
    )
