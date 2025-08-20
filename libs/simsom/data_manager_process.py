"""
The data manager is responsible for choosing Users to run, save on disk generated data and
"""

import random as rnd
from mpi4py import MPI
from mpi_utils import iprobe_with_timeout, handle_crash, gettimestamp
import os


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

    print(f"[{gettimestamp()}] DataMngr (PID: {os.getpid()}) >> running...", flush=True)
    print(f"[{gettimestamp()}] DataMngr >> network size: {len(users)}", flush=True)

    # Arch status object
    status = MPI.Status()

    # Outgoing messages
    outgoing_messages = {user.uid: [] for user in users}
    outgoing_passivities = {user.uid: [] for user in users}

    # User objects main structure
    users_dict = {}
    for u in users:
        users_dict[u.uid] = u

    # All actions
    firehose_buffer = []

    # Clock
    clock = ClockManager()

    # Manage user selection
    selected_users = set()

    alive = True

    # Bootstrap sync
    comm_world.barrier()

    while True:

        if iprobe_with_timeout(comm_world=comm_world, status=status, pname="DataMngr"):

            sender, payload = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

            # Check if termination signal has been sent
            if alive and payload == "STOP":
                print(
                    f"[{gettimestamp()}] DataMngr >> stop signal detected!", flush=True
                )
                alive = False

            if alive:

                if sender == "worker":

                    # print(
                    #     f"[{gettimestamp()}] DataMngr > processed user batch received...",
                    #     flush=True,
                    # )

                    firehose_chunk = []

                    for processed_user_pack in payload:

                        # Unpack the agent + incoming messages and passive actions
                        user, new_msgs, passive_actions = processed_user_pack

                        # Assign a timestamp
                        for msg in new_msgs:
                            msg.time = clock.next_time()  # type: ignore
                            firehose_chunk.append(msg)

                        # Updating main structures
                        outgoing_messages[user.uid].extend(new_msgs)  # type: ignore
                        outgoing_passivities[user.uid].extend(passive_actions)  # type: ignore

                        # Updating user object
                        users_dict[user.uid] = user  # type: ignore

                    firehose_buffer.append(firehose_chunk)

                elif sender == "recommender_system":

                    users_pack_batch = []

                    # Since we risk to shuffle the users when we build the batch, we need to
                    # make sure we don't pick the same user twice
                    batch_size = min(batch_size, len(users) - len(selected_users))
                    # print(f"[{gettimestamp()}] DataMngr > {len(users)}", flush=True)
                    # print(
                    #     f"[{gettimestamp()}] DataMngr > {len(selected_users)}",
                    #     flush=True,
                    # )
                    # print(f"[{gettimestamp()}] DataMngr > {batch_size}", flush=True)

                    # Build the batch
                    for _ in range(batch_size):
                        # Always pick the first user (round-robin style)
                        picked_user = users_dict[users[0].uid]

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
                        users_pack_batch.append(
                            (picked_user, active_actions_send, passive_actions_send)
                        )

                        # TODO: Flush outgoing messages ????
                        outgoing_messages[picked_user.uid] = []
                        outgoing_passivities[picked_user.uid] = []

                        # Before we process all the users, we need to shuffle them
                        if len(selected_users) == len(users):
                            rnd.shuffle(users)
                            selected_users.clear()
                            # print(f"[{gettimestamp()}] DataMngr: user reset", flush=True)

                    # Firehose data
                    firehose_flush = []
                    if len(firehose_buffer) > 0:
                        firehose_flush = firehose_buffer.pop(0)

                    # print(
                    #     f"[{gettimestamp()}] DataMngr >> sending {len(firehose_flush)} messages",
                    #     flush=True,
                    # )

                    # print(
                    #     f"[{gettimestamp()}] DataMngr >> firehose size: {len(firehose)}",
                    #     flush=True,
                    # )

                    comm_world.send(
                        ("data_manager", (users_pack_batch, firehose_flush)),
                        dest=rank_index["recommender_system"],
                    )

                elif sender == "policy_evaluator":

                    # print(
                    #     f"[{gettimestamp()}] DataMngr >> data from policy evaluator",
                    #     flush=True,
                    # )
                    # Get the moderated user/content info and apply logic to data
                    continue

                else:

                    print(
                        f"[{gettimestamp()}] DataMngr >> unknown sender: {sender}",
                        flush=True,
                    )
                    raise ValueError

        else:

            print(f"[{gettimestamp()}] DataMngr >> closing...", flush=True)

            if alive:

                handle_crash(
                    comm_world=comm_world,
                    status=status,
                    srank=rank,
                    srole="data_manager",
                    pname="DataMngr",
                )

            print(f"[{gettimestamp()}] DataMngr >> entering barrier...", flush=True)
            comm_world.barrier()
            break

    print(f"[{gettimestamp()}] DataMngr >> closed.", flush=True)

    print(f"[{gettimestamp()}] DataMngr >> final clock: {clock.current_time} ")
