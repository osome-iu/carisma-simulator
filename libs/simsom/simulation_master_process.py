"""
Simulation master process module.
"""

import random
from simtools import get_scores_below_threshold
from mpi4py import MPI
import numpy as np


class ClockManager:
    """
    Class responsible for clock simulation,
    the class has the task of giving a value obtained from a distribution.
    """

    def __init__(self) -> None:
        self.current_time = 0

    def next_time(self) -> int:
        """
        Return the current time and generate the next
        Returns:
            int: current time
        """
        current = self.current_time
        # TODO: find a distribution for this
        self.current_time += random.random() * 0.02
        return current


def propagate_message(user_index: int, users: list) -> None:
    """
    Function to propagate the message of a user.
    Since we cannot modifiy feeds of users into the class User

    Args:
        user_index (int): graph index of the focal user that's doing the message propagation
        users (list): lists of all users
    """

    target = users[user_index]
    for friend_uid in target.friends:
        users[friend_uid].user_feed.appendleft(target.last_message)


def simulation_master(
    comm: MPI.Intracomm,
    size: int,
    users: list,
    max_iteration: int,
    sigterm: str = "STOP",
) -> None:
    """
    Agent_process_manager (Process 0) is the most important process of the simulator.
    It takes care of agent_process processes to be executed
    sending to them users_id for data generation.
    It takes care of shutting down the processes (to date) and to communicate with
    policy to enforce the policies that will be implemented.

    Args:
        comm (MPI.COMM_WORLD): communication context between processes
        size (int): number of processes running in the simulation
        users (list): list of users
        max_iteration (int): max number of iteration that the simulation does before it stops
    """

    # DEBUG
    banned = set()
    count = 0
    clock = ClockManager()

    index = 0
    # get M-4 user_ids, send the name of the user user_ids[index] to the corresponding process
    for i in range(4, size):
        comm.send(users[index], dest=i)
        index += 1
        index = index % len(users)

    while True:
        # send termination signal when all the processes are done.
        if max_iteration > 0:
            if count == max_iteration:
                # !! Do not change this, sigterm needs to be sent in the correct order
                for i in range(4, size):
                    comm.send(sigterm, dest=i)
                comm.send(sigterm, dest=1)
                comm.send(sigterm, dest=2)
                break
        else:
            status = MPI.Status()
            flag = comm.Iprobe(source=0, status=status)
            if flag:
                data = np.empty(1, dtype="i")
                req = comm.Irecv(data, source=0)  # non-blocking receive
                req.Wait()
                # !! Do not change this, sigterm needs to be sent in the correct order
                for i in range(4, size):
                    comm.send(sigterm, dest=i)
                comm.send(sigterm, dest=1)
                comm.send(sigterm, dest=2)
                break

        # Blocking receive instead of waitany
        user, real_rank = comm.recv(source=MPI.ANY_SOURCE)
        user.add_clock(clock.next_time())
        # example of how a policy could work:
        # send a request to process 0 (policy)
        # comm.send(user, dest=0)
        # status = comm.recv(source=0)
        status = get_scores_below_threshold(user, 0)

        if status:
            user.is_suspended = True
            # print(f"USER {user.user_id} WITH AT LEAST 3 LOW QUALITY CONTENTS")
        # we put the modified object into users list
        # TODO: ensure the user object is updated in the users list

        users[user.user_index] = user
        propagate_message(user_index=user.user_index, users=users)
        # send the user action to process 1 (update queue manager)
        comm.send(user.last_message, dest=2)
        # trigger the process to generate a new action
        while users[index].is_suspended:
            banned.add(users[index].user_id)
            index += 1
            index = index % len(users)
            print(f"User {user.user_id} is banned")
        comm.send(users[index], dest=real_rank)
        index += 1
        index = index % len(users)
        count += 1
