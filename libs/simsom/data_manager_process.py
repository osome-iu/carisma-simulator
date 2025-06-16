"""
The data manager is responsible for choosing Users to run, save on disk generated data and
"""

import random as rnd
from mpi4py import MPI
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


def flush_incoming_messages(comm, status):
    while comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
        _ = comm.recv(source=status.Get_source(), tag=status.Get_tag(), status=status)


class ClockManager:
    """
    Class responsible for clock simulation,
    the class has the task of giving a value obtained from a distribution.
    """

    def __init__(self) -> None:
        self.current_time = 0

    def next_time(self):
        """
        Return the current time and generate the next
        Returns:
            int: current time
        """
        current = self.current_time
        # TODO: find a distribution for this
        self.current_time += rnd.random() * 0.02

        return current


def run_data_manager(
    users,
    comm_world: MPI.Intracomm,
    rank: int,
    size: int,
    rank_index: dict,
    batch_size=5,
):

    print("* Data manager >> running...", flush=True)

    # Arch status object
    status = MPI.Status()

    # Outgoing messages
    outgoing_messages = {user.uid: [] for user in users}
    outgoing_passivities = {user.uid: [] for user in users}

    # Clock
    clock = ClockManager()

    # Manage user selection
    selected_users = set()

    # Bootstrap sync
    comm_world.barrier()

    while True:

        data = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

        # Check if termination signal has been sent
        if status.Get_tag() == 99:
            print("DataMngr >> (1) sigterm detected, entering barrier...", flush=True)
            flush_incoming_messages(comm_world, status)
            comm_world.barrier()
            break

        sender, content = data

        if sender == "agent_proc":

            # Unpack the agent + incoming messages and passive actions
            user, new_msgs, passive_actions = content

            # Assign a timestamp
            for sender in new_msgs:
                sender.time = clock.next_time()

            # print(f"* Data manager >> {user.uid} has {len(new_msgs)} new messages", flush=True)
            # print(f"* Data manager >> {user.uid} has {len(passive_actions)} new passivities", flush=True)

            outgoing_messages[user.uid].extend(new_msgs)
            outgoing_passivities[user.uid].extend(passive_actions)

        elif sender == "recsys_proc":

            users_packs_batch = []

            # Since we risk to shuffle the users when we build the batch, we need to
            # make sure we don't pick the same user twice
            batch_size = min(batch_size, len(users) - len(selected_users))

            # Build the batch
            for _ in range(batch_size):
                # Always pick the first user (round-robin style)
                picked_user = users[0]

                # Track selected user
                selected_users.add(picked_user.uid)
                # NOTE: A set could be avoided

                # Move picked user to the end of the list
                users = users[1:] + [picked_user]
                # NOTE: Can be optimized

                # Get the in and out messages based on friends
                active_actions_send = outgoing_messages[picked_user.uid]
                passive_actions_send = outgoing_passivities[picked_user.uid]

                # Add it to the batch
                users_packs_batch.append(
                    (picked_user, active_actions_send, passive_actions_send)
                )

                # TODO: Flush outgoing messages ????
                outgoing_messages[picked_user.uid] = []
                outgoing_passivities[picked_user.uid] = []

                # Before we process all the users, we need to shuffle them
                if len(selected_users) == len(users):
                    rnd.shuffle(users)
                    selected_users.clear()

            req1 = comm_world.isend(
                users_packs_batch, dest=rank_index["recommender_system"]
            )

            safe_finalize_isends([req1])

        elif sender == "policy_proc":

            print("* Data manager >> data from policy", flush=True)
            # Get the moderated user/content info and apply logic to data
            continue

        # elif sender == "analyzer_proc" and content == "STOP":

        #     print("* Data manager >> Stopping simulation...")

        #     # # Flush pending incoming messages
        #     # while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
        #     #     _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

        #     comm_world.Barrier()
        #     break

        else:

            print("* Data manager >> unknown message", flush=True)
            raise ValueError

    print("* Data manager >> closed.", flush=True)
