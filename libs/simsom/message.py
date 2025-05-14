"""class for a message.
This class allow us to have the message object that is put inside
user's feed.
Messages are object used in the users.py as vector of contents.
"""

import random
import numpy as np
from action import Action


class Message(Action):
    def __init__(
        self,
        mid: int,
        uid: int,
        quality_params: tuple,
        topics: list,
        is_shadow: bool,
        exposure: list = [],
    ) -> None:
        Action.__init__(self, mid, uid)
        self.quality_params = quality_params
        self.topics = topics
        self.is_shadow = is_shadow
        self.exposure = exposure
        self.appeal = self.appeal_func()
        if quality_params:
            self.quality = self.custom_beta_quality(self.quality_params)
        else:
            self.quality = None
        self.time = None
        self.reshared_id = np.nan
        self.reshared_original_id = np.nan
        self.reshared_user_id = np.nan

    def expon_quality(self, lambda_quality=-5) -> float:
        """return a quality value x via inverse transform sampling
        Pdf of quality: $f(x) sim Ce^{-lambda x}$, 0<=x<=1
        $C = frac{lambda}{1-e^{-lambda}}$

        Args:
            lambda_quality (int, optional): Defaults to -5.

        Returns:
            float: quality value
        """
        x = random.random()
        return np.log(1 - x + x * np.e ** (-1 * lambda_quality)) / (-1 * lambda_quality)

    def custom_beta_quality(self, distribution_param: tuple) -> float:
        """return a quality value x via beta distribution with alpha and beta params
        Since we use a custom beta distribution version, we have limits within
        which we want our values to come out.
        If, for example, values between 0 and 0.3 are set, then we discard values
        greater than 0.3 and take the first one that falls within the set range.

        Args:
            distribution_param (tuple): tuple to define the value of
            alpha - beta for the distribution
            and lower - upper bound for the value of the quality

        Returns:
            float: quality value
        """
        if distribution_param:
            alpha, beta, lower, upper = distribution_param
            checked = False
            while not checked:
                quality = round(np.random.beta(alpha, beta), 2)
                if quality >= lower and quality <= upper:
                    checked = True
            return quality
        else:
            return self.expon_quality()

    def appeal_func(self, exponent=5) -> float:
        """
        Return an appeal value a following a right-skewed distribution
        via inverse transform sampling
        Pdf of appeal: $P(a) = (1+alpha)(1-a)^{alpha}$
        exponent = alpha+1 characterizes the rarity of high appeal values
        --- the larger alpha, the more skewed the distribution
        """
        # if the users that post the message are under shadowban
        # the appeal should be 0 to not be reshared
        if self.is_shadow:
            return 0
        u = random.random()
        return 1 - (1 - u) ** (1 / exponent)

    def assign_clock(self, time: float) -> None:
        """just assign the clock value to the message

        Args:
            time (float): time of the clock
        """
        self.time = time

    def __str__(self) -> str:
        return "\n".join(
            [
                f"- Message id: {self.aid}",
                f"- User id: {self.uid}",
                f"- Quality parameters: {self.quality_params}",
                f"- Quality: {self.quality}",
                f"- Time: {self.time}",
                f"- Topics: {self.topics}",
            ]
        )

    def write_action(self):
        """function to write active action (post, repost) to the disk

        Returns:
            tuple: return the values that we want to keep on the disk
        """
        parent_values = super().write_action()
        return (
            *parent_values,
            self.quality,
            self.appeal,
            self.reshared_id,
            self.reshared_user_id,
            self.reshared_original_id,
            self.time,
        )
