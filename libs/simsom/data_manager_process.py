"""
The data manager is responsible for choosing Users to run, save on disk generated data and
"""

import random as rnd
from mpi4py import MPI
from mpi_utils import iprobe_with_timeout, clean_termination, handle_crash


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
    print(f"* Data manager >> network size: {len(users)}", flush=True)

    # Arch status object
    status = MPI.Status()

    # Outgoing messages
    outgoing_messages = {user.uid: [] for user in users}
    outgoing_passivities = {user.uid: [] for user in users}

    # Clock
    clock = ClockManager()

    # Manage user selection
    selected_users = set()

    alive = True

    isends = []

    # Worker process ranks
    worker_ranks = list(range(5, size))

    # Bootstrap sync
    comm_world.barrier()

    while True:

        if iprobe_with_timeout(
            comm_world,
            source=MPI.ANY_SOURCE,
            tag=MPI.ANY_TAG,
            status=status,
            pname="DataMngr",
        ):

            sender, payload = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

            # Check if termination signal has been sent
            if sender == "analyzer" and payload == "STOP" and alive:
                print("* DataMngr >> stop signal detected", flush=True)
                alive = False
            elif payload == "STOP" and alive:
                print("* DataMngr >> crashing...", flush=True)
                alive = False

            # Wait for pending isends
            MPI.Request.waitall(isends)
            isends.clear()

            if alive:

                if sender == "worker":

                    # LOGIC:
                    # Once a single worker process sends a processed user,
                    # triggers a scan to possibly collect other processed users.
                    # The scan look at all worker processes quickly.
                    # If some users is not collected would be in the very next.
                    # It's a way to give priority to worker processes.

                    rnd.shuffle(worker_ranks)

                    for source in worker_ranks:

                        # Unpack the agent + incoming messages and passive actions
                        user, new_msgs, passive_actions = payload

                        # Assign a timestamp
                        for msg in new_msgs:
                            msg.time = clock.next_time()  # type: ignore

                        # Updating main structures
                        outgoing_messages[user.uid].extend(new_msgs)  # type: ignore
                        outgoing_passivities[user.uid].extend(passive_actions)  # type: ignore

                        # Scan for incoming processed user from current source worker
                        if comm_world.iprobe(source=source, status=status):
                            # If new processed user available receive and update content
                            _, payload = comm_world.recv(source=source, status=status)

                elif sender == "recsys" and payload == "dataReq":

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

                    isends.append(
                        comm_world.isend(
                            ("dataMngr", users_packs_batch),
                            dest=rank_index["recommender_system"],
                        )
                    )

                elif sender == "policyMngr":

                    print("* Data manager >> data from policy", flush=True)
                    # Get the moderated user/content info and apply logic to data
                    continue

                else:

                    print("* Data manager >> unknown sender", flush=True)
                    raise ValueError

        else:

            print(f"* DataMngr >> closing with {len(isends)} isends...", flush=True)

            if alive:

                handle_crash(
                    comm_world=comm_world,
                    status=status,
                    srank=rank,
                    srole="data_manager",
                    pname="DataMngr",
                )

            print("* DataMngr >> entering barrier...", flush=True)
            comm_world.barrier()
            break

    print("* DataMngr >> closed.", flush=True)

    print(f"* DataMngr >> final clock: {clock.current_time} ")
