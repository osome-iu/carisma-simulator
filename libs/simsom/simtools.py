"""
Helper and utilities module.
Put here function that can be used by more than one process.
"""

import os
import csv
from user import User
import random
import igraph as ig

MINIMUM_REQUIRED_ATTRIBS = {"uid", "utype", "postperday", "qualitydistr"}
QUALITYDISTR = "(0.5, 0.15, 0, 1)"


def read_empirical_network(file):
    """
    Read a network from file path.
    """
    try:
        raw_net = ig.Graph.Read_GML(file)

        # prevent errors with duplicate attribs
        net = _delete_unused_attributes(
            raw_net, desire_attribs=MINIMUM_REQUIRED_ATTRIBS
        )
    except Exception as e:
        print("Exception when reading network")
        print(e.args)
    return net


def _delete_unused_attributes(net, desire_attribs):
    for attrib in net.vs.attributes():
        if attrib not in desire_attribs:
            del net.vs[attrib]
    return net


def init_network(file=None, net_size=200, p=0.5, k_out=3) -> dict:
    """
    Create a network using a directed variant of the random-walk growth model
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.67.056104
    Inputs:
        - net_size (int): number of nodes in the desired network
        - k_out (int): average no. friends for each new node
        - p (float): probability for a new node to follow friends of a
        friend (models network clustering)
    """
    if file:
        graph = read_empirical_network(file)
    else:
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
            friends += random.sample(
                range(graph.vcount()), k_out - 1 - n_random_friends
            )
            graph.add_vertex(n)
            edges = [(n, f) for f in friends]
            graph.add_edges(edges)
        for v in graph.vs:
            v["uid"] = f"u{v.index}"
            # v["utype"] = random.choice(["lurker", "normal user"])
            v["utype"] = "normal user"
            v["postperday"] = 0 if v["utype"] == "lurker" else random.uniform(0, 50)
            v["qualitydistr"] = QUALITYDISTR

    users = []
    for node in graph.vs:
        # remember link direction is following
        friends = graph.successors(node.index)  # this returns the index of the vertex
        followers = graph.predecessors(node.index)
        user_i = User(
            uid=graph.vs[node.index]["uid"],
            user_class=graph.vs[node.index]["utype"],
            post_per_day=int(graph.vs[node.index]["postperday"]),
            quality_params=eval(graph.vs[node.index]["qualitydistr"]),
            friends=friends,
            followers=followers,
        )
        users.append(user_i)
    return users
