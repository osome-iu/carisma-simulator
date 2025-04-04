import igraph as ig
import random
from user import User
import numpy as np


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
            friends=["u" + str(u) for u in friends],
            followers=["u" + str(u) for u in followers],
        )
        users[user_i.uid] = user_i
    return users


class DataManager:

    def __init__(self):
        self.agents = init_network(20)
        self.outgoing_msgs = {uid: [] for uid in self.agents.keys()}

    def recv_from_agents(self) -> None:
        for agent in self.agents.values():
            act, pact = agent.make_actions()
            self.outgoing_msgs[agent.uid].extend(act)
            # TODO: handle the passive actions (pact)

    def send_recsys(self, user: User) -> tuple:
        in_messages = []
        out_messages = []
        for user_id, outgoing_messages in self.outgoing_msgs.items():
            if user_id in user.followers:
                in_messages.extend(outgoing_messages)
            else:
                out_messages.extend(outgoing_messages)
        return in_messages, out_messages


class RecSys:
    def __init__(self):
        self.feeds = {}

    def build_feed(
        self,
        agent: User,
        in_messages: list,
        out_messages: list,
        in_perc: float = 0.5,
        out_perc: float = 0.5,
    ) -> list:
        random.shuffle(in_messages)
        random.shuffle(out_messages)
        n_in = int(len(in_messages) * in_perc)
        n_out = int(len(out_messages) * out_perc)
        new_feed = in_messages[:n_in] + out_messages[:n_out]
        random.shuffle(new_feed)
        if len(new_feed) > agent.cut_off:
            new_feed = new_feed[: agent.cut_off]
        self.feeds[agent.uid] = new_feed
        agent.newsfeed = self.feeds[agent.uid]
        return agent.newsfeed


class ConvergenceMonitor:
    def __init__(self, threshold: float = 0.001):
        self.threshold = threshold
        self.data = []
        self.prev_mean = None

    def get_data_from_recsys(self, newsfeed: list, convergence_size: int = 1000):
        self.data.extend(newsfeed)
        if len(self.data) >= convergence_size:
            self.check_convergence()

    def check_convergence(self):
        mean = np.mean([message.quality for message in self.data])
        if self.prev_mean:
            diff = abs(mean - self.prev_mean)
            print(diff)
            if diff < self.threshold:
                return False
            else:
                self.data = []
                self.prev_mean = mean
                return True
        else:
            self.data = []
            self.prev_mean = mean
            return True


if __name__ == "__main__":

    dm = DataManager()
    rs = RecSys()
    cm = ConvergenceMonitor()
    condition = True

    while condition:
        dm.recv_from_agents()
        for agent in dm.agents.values():
            print("Processing agent: ", agent.uid)
            in_messages, out_messages = dm.send_recsys(agent)
            newsfeed = rs.build_feed(agent, in_messages, out_messages)
            cm.get_data_from_recsys(newsfeed)
            condition = cm.check_convergence()
            if not condition:
                print("Convergence reached!")
                break
