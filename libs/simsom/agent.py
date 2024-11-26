"""
Stub class for User/Agent objects.
Original code has to be ported.
"""

import random as rnd
from message import Message


class Agent:
    def __init__(self, aid):
        self.aid = aid
        self.followers = []
        self.newsfeed = []
        self.post_counter = 0
        self.repost_counter = 0

    def make_actions(self, min_actions=0, max_actions=4, cut_off=15):
        """
        Perform an action that could be a post or a repost.
        TODO: remove min, max, port from old code
        Args:

        Returns:
            list of Message obj: the list of messages that the agent produced
        """
        actions = []

        for _ in range(rnd.randint(min_actions, max_actions)):
            if self.newsfeed and rnd.random() < 0.8:
                repost = rnd.choice(self.newsfeed)
                new_msg = Message(
                    mid=f"{repost.mid}_repost_{self.aid}_{self.repost_counter}"
                )
                actions.append(new_msg)
                self.repost_counter += 1
            else:
                new_msg = Message(mid=f"{self.aid}_post_{self.post_counter}")
                actions.append(new_msg)
                self.post_counter += 1
        self.newsfeed = self.newsfeed[:cut_off]

        return actions
