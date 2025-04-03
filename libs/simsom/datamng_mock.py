import igraph as ig
import random
from user import User
from tqdm import tqdm
import time


def init_network(net_size=200, p=0.5, k_out=3) -> dict:
    """
    Create a network using a directed variant of the random-walk growth model
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.67.056104
    Inputs:
        - net_size (int): number of nodes in the desired network
        - k_out (int): average no. friends for each new node
        - p (float): probability for a new node to follow friends of a
        friend (models network clustering)
    """
    # if file:
    #     graph = read_empirical_network(file)
    # else:
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
    for v in graph.vs:
        v["uid"] = f"u{v.index}"
        # v["utype"] = random.choice(["lurker", "normal user"])
        v["utype"] = "normal user"
        v["postperday"] = 0 if v["utype"] == "lurker" else random.uniform(0, 50)
        v["qualitydistr"] = "(0.5, 0.15, 0, 1)"  # QUALITYDISTR

    users = {}
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
        users[user_i.uid] = user_i
    return users


class DataManager:

    def __init__(self):
        self.agents = init_network()
        self.outgoing_msgs = {uid: [] for uid in self.agents.keys()}

    def recv_from_agents(self):
        for agent in tqdm(self.agents.values()):
            act, pact = agent.make_actions()
            self.outgoing_msgs[agent.uid].extend(act)
            # TODO: handle the passive actions (pact)

    def recsys_request():

        pass

    def run_dmng(self):

        while True:
            print("Sleeping... (msg prodcution)")
            time.sleep(2)
            self.recv_from_agents()
            break


if __name__ == "__main__":

    dm = DataManager()
    dm.run_dmng()
