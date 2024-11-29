"""
The data manager is responsible for choosing Users to run, save on disk generated data and 
"""

import time
import os
import csv
import random as rnd
import copy
import numpy as np
from mpi4py import MPI
from user import User

time_now = int(time.time())
folder_path = f"files/{time_now}"
file_path_activity = folder_path + "/activities.csv"
file_path_passivity = folder_path + "/passivities.csv"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
if os.path.isfile(file_path_activity):
    os.remove(file_path_activity)
if os.path.isfile(file_path_passivity):
    os.remove(file_path_passivity)
with open(file_path_activity, "w", newline="", encoding="utf-8") as out_act:
    csv_out_act = csv.writer(out_act)
    if os.stat(file_path_activity).st_size == 0:
        csv_out_act.writerow(
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

with open(file_path_passivity, "w", newline="", encoding="utf-8") as out_pas:
    csv_out_pas = csv.writer(out_pas)
    if os.stat(file_path_passivity).st_size == 0:
        csv_out_pas.writerow(
            [
                "user_id",
                "action_id",
                "message_id",
                "message_user_id",
            ]
        )


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
        self.current_time += rnd.random() * 0.02
        return current


def batch_message_propagation(message_board: dict, user: User, messages: list):
    # TODO: pls explain
    # Explanation: this code is a reminder for the high follower spreading problem (this implementation is a stub).
    # As the simulation size grow, users with lots of follower (power law) can require lot of time for spreading.
    batch_size = 1000
    for i in range(0, len(user.followers), batch_size):
        follower_batch = user.followers[i : i + batch_size]
        for m in messages:
            for follower_uid in follower_batch:
                message_board["u" + str(follower_uid)].append(copy.deepcopy(m))
    # NOTE: an equivalent (and more simple) implementation could be:
    # for follower_aid in agent.followers:
    #     for m in messages:
    #         message_board[follower_aid].append(copy.deepcopy(m))


def run_data_manager(
    users: list,
    message_count_target: int,
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,
    rank_index: dict,
):
    """
    We're confusion
    - b_size: batch size? # Yes
    - message_count_target # The target number of message to generate for a run (is to have a stopping criterion)
    """
    # TODO: move initialization to `simsom` (probably)
    # Example sim params
    batch_size = 5

    # Followers
    incoming_messages = {user.uid: [] for user in users}

    # Status of the processes
    status = MPI.Status()

    # Clock
    clock = ClockManager()

    # DEBUG #
    # msgs_store = []

    # Bootstrap sync
    comm_world.Barrier()

    print(f"Data manager start @ rank: {rank}")

    # Batch processing
    message_count = 0
    while True:
        if message_count_target == 0:
            flag = comm_world.Iprobe(
                source=rank_index["convergence_monitor"], status=status
            )
            if flag:
                data = np.empty(1, dtype="i")
                req = comm_world.Irecv(
                    data, source=rank_index["convergence_monitor"]
                )  # non-blocking receive
                req.Wait()
                comm_world.send("sigterm", dest=rank_index["policy_filter"])
                break
        else:
            if message_count >= message_count_target:
                break
        batch_send_req = None

        # Unpicked agents count
        n_users = len(users)
        # TODO: how about n_agents < b_size ? # It's unlikely but in that case b_size should be resized to n_agents
        if n_users >= batch_size:

            users_packs_batch = []

            # Build the batch
            for i in range(batch_size):

                # Pick an agent at random (without replacement)
                user_index = rnd.choice(range(n_users - i))
                picked_user = users.pop(user_index)
                # Get the incoming messages and pack
                user_pack = (picked_user, incoming_messages[picked_user.uid])
                # Add it to the batch
                users_packs_batch.append(user_pack)
                # Flush incoming messages
                incoming_messages[picked_user.uid] = []

            # Non blocking send the batch to the policy_filter
            batch_send_req = comm_world.isend(
                users_packs_batch,
                dest=rank_index["policy_filter"],
            )

        # Handlers harvesting
        returned_users = 0
        # TODO: pls explain this condition
        # Explanation: here we are gathering processed agents from the agent processes.
        # While you don't have exactly the same amount of agents you sent in the batch,
        # the loop won't stop to collect agents. This ensure to not empty the agent structure.
        while returned_users < batch_size:

            # Scan once all the handlers for an agent that completed
            for source in range(rank_index["agent_handler"], size):

                # Check if a handler has done and is waiting
                if comm_world.iprobe(source=source, status=status):

                    # Collect and unpack modified agent and actions
                    mod_user, new_msgs, passive_actions = comm_world.recv(
                        source=source,
                        status=status,
                    )
                    for msg in new_msgs:
                        msg.time = clock.next_time()

                    # Dispatch the messages to agent followers
                    batch_message_propagation(
                        incoming_messages,
                        mod_user,
                        new_msgs,
                    )

                    # Put the agent back
                    users.append(mod_user)
                    returned_users += 1

                    # Increase counter by the number of action (messages) produced
                    message_count += len(new_msgs)

                    # DEBUG #
                    # csv_out_act.writerow(m.write_action())
                    with open(
                        file_path_activity, "a", newline="", encoding="utf-8"
                    ) as out_act:
                        csv_out_act = csv.writer(out_act)
                        for m in new_msgs:
                            csv_out_act.writerow(m.write_action())
                        # csv_out_pas.writerow(a.write_action())
                    with open(
                        file_path_passivity, "a", newline="", encoding="utf-8"
                    ) as out_pas:
                        csv_out_pas = csv.writer(out_pas)
                        for a in passive_actions:
                            csv_out_pas.writerow(a.write_action())
                    # msgs_store.append(
                    #     (
                    #         m.time,
                    #         f"User {mod_user.uid}, Created: {m.mid} message",
                    #     )
                    # )
                    # NOTE: I'm storing all the messages produced paired with the real creation timestamp
                    # TODO: change this to distribution time

        # Check if batch correctly transmitted
        if batch_send_req:
            batch_send_req.wait()

    # Close policy filter before quitting
    comm_world.send("sigterm", dest=rank_index["policy_filter"])

    print(f"Data manager stop @ rank: {rank}")

    # DEBUG #
    # msgs_store.sort(key=lambda x: x[0])  # sort by real timestamp
    # for msg in msgs_store:
    #     print(f"[{msg[0]}] - {msg[1]}")
