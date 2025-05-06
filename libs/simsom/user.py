"""
Stub class for User/Agent objects.
Original code has to be ported.
"""

from message import Message
import random
import pandas as pd
from view import View


class User:
    def __init__(
        self,
        uid: str,
        user_class: str,
        quality_params: str,
        post_per_day: float,
        friends: list = [],
        followers: list = [],
    ):
        self.uid = uid
        self.followers = followers
        self.friends = friends
        self.user_class = user_class
        self.post_per_day = post_per_day
        self.quality_params = quality_params
        self.cut_off = 15
        if self.post_per_day > 15:
            self.cut_off = self.post_per_day
        self.newsfeed = []
        self.post_counter = 0
        self.repost_counter = 0
        self.view_counter = 0
        user_description = []
        for _ in range(random.randrange(1, 6)):
            user_description.append(random.randrange(0, 5))
        self.user_description = list(set(user_description))
        self.is_suspended = False
        self.is_shadow = False
        self.mu = 0.5

    def make_actions(self) -> None:
        """function that is responsible for routing the action between new action and re-sharing.
        The choice is made according to the mu parameter and the user's feed, which must contain at least one message

        Args:
            message_id (str): message_id from the main_sim, is a combination of
            user_id and a specific index
            is_shadow (bool): flag to check if the user is under shadow-ban
            quality_params (tuple, optional): params to get the quality from the beta.
            Defaults to (0.5, 0.15, 0, 1).
        """
        actions = []
        passive_actions = []
        for _ in range(self.post_per_day):
            if len(self.newsfeed) > 0 and random.random() > self.mu:
                passive_action, active_action = self.reshare_message()
                actions.append(active_action)
                passive_actions.extend(passive_action)
            else:
                actions.append(self.post_message())
        self.newsfeed = self.newsfeed[: self.cut_off]
        return actions, passive_actions

    def reshare_message(self) -> None:
        """function to reshare a message, a message is chosen at random within the user's feed.
        A new message is created, taking the attributes (appeal and quality) of the old message,
        the first reshare and the following parents (id and user_id) are kept track of.

        Args:
            message_id (str): message_id from the main_sim, is a combination of
            user_id and a specific index
            is_shadow (bool): flag to check if the user is under shadow-ban
        """
        # target = random.sample(list(self.user_feed), 1)[0]
        target = None
        appeal_threshold = random.random()
        passive_actions = []
        for msg in self.newsfeed:
            vid = "V" + str(self.view_counter) + "_" + self.uid
            v = View(vid=vid, uid=self.uid, parent_mid=msg.aid, parent_uid=msg.uid)
            passive_actions.append(v)
            self.view_counter += 1
            if msg.appeal >= appeal_threshold:
                target = msg
        if not target:
            target = random.sample(list(self.newsfeed), 1)[0]
        message_reshared = Message(
            mid="R" + str(self.repost_counter) + "_" + str(self.uid),
            uid=self.uid,
            quality_params=None,
            topic=target.topic,
            is_shadow=self.is_shadow,
            exposure=target.exposure,
        )
        message_reshared.quality = target.quality
        message_reshared.appeal = target.appeal
        # If it's not the first reshare we get the attributes
        if pd.notna(target.reshared_id):
            message_reshared.reshared_original_id = target.reshared_original_id
            message_reshared.reshared_id = target.aid
        else:
            # If it's the first reshare reshared_original_id and reshared_id are the same
            message_reshared.reshared_id = target.aid
            message_reshared.reshared_original_id = target.aid
        message_reshared.reshared_user_id = target.uid
        # self.reshared_messages.append(message_reshared)
        self.repost_counter += 1
        return passive_actions, message_reshared

    def post_message(self) -> None:
        """function to post a message, since the feed of the user
        has a specific size (e.g. 10) the module is used to place the
        message in a specific position

        Args:
            message_id (str): message_id from the main_sim, is a combination of
            user_id and a specific index
            is_shadow (bool): flag to check if the user is under shadow-ban
            quality_params (tuple, optional): params to get the quality from the beta.
            Defaults to (0.5, 0.15, 0, 1).
        """
        message_created = Message(
            mid="P" + str(self.post_counter) + "_" + str(self.uid),
            uid=self.uid,
            topic=random.choice(self.user_description),
            is_shadow=self.is_shadow,
            quality_params=self.quality_params,
        )
        # self.shared_messages.append(message_created)
        self.post_counter += 1
        return message_created

    def __str__(self) -> str:
        return "\n".join(
            [
                f"User id: {self.uid}",
                f"- User type: {self.user_class}",
                f"- Post per day: {self.post_per_day}",
                f"- Number of post: {self.post_counter}",
                f"- Number of repost: {self.repost_counter}",
                f"- Shadow status:  {self.is_shadow}",
                f"- Suspension status: {self.is_suspended}",
                f"- Feed: {self.newsfeed}",
                f"- Friends: {self.friends}",
                f"- Followers: {self.followers}",
                f"- Description: {self.user_description}",
            ]
        )
