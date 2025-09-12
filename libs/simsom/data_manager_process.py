"""
The data manager is responsible for choosing Users to run, save on disk generated data and
"""

import os
import numpy as np
import random as rnd
import math
from mpi4py import MPI
from collections import deque
from mpi_utils import iprobe_with_timeout, handle_crash, gettimestamp
from clock_manager import ClockManager

# class ClockManager:
#     """
#     Class responsible for clock simulation,
#     the class has the task of giving a value obtained from a distribution.
#     """

#     def __init__(self) -> None:
#         self.current_time = 0

#     def next_time(self):
#         """
#         Return the current time and generate the next
#         Returns:
#             int: current time
#         """
#         current = self.current_time
#         # TODO: find a distribution for this
#         # self.current_time += rnd.random() * 0.0003
#         delta = np.abs(np.random.normal(0.0003, 0.0005))
#         self.current_time += delta

#         return current

class UserActivitySimulator:
    def __init__(
            self, mean_daily_actions, 
            transition_matrix=None,
            state_multipliers=(0.0, 0.5, 1.0, 2.5)
    ):

        self.mean_daily_actions = np.array(mean_daily_actions)
        self.n_users = len(mean_daily_actions)
        self.n_states = len(state_multipliers)
        self.state_multipliers = np.array(state_multipliers)
        
        if transition_matrix is None:
            # matrice con diagonale alta → persistenza dello stato
            base = np.array([
                [0.85, 0.10, 0.05, 0.00],  # da inattivo
                [0.10, 0.70, 0.15, 0.05],  # da low
                [0.05, 0.10, 0.70, 0.15],  # da normal
                [0.05, 0.05, 0.20, 0.70],  # da burst
            ])
            self.transition_matrix = base
        else:
            self.transition_matrix = np.array(transition_matrix)
        
        # stato iniziale: quasi tutti inattivi
        self.state = np.random.choice(
            self.n_states, 
            size=self.n_users, 
            p=[0.7, 0.2, 0.09, 0.01]
        )
    
    def step(self):
        """
        Simulate a new day actions number.
        """
        new_state = np.zeros_like(self.state)
        actions = np.zeros(self.n_users, dtype=int)
        
        for i in range(self.n_users):
            # evoluzione stato secondo Markov
            current = self.state[i]
            probs = self.transition_matrix[current]
            new_state[i] = np.random.choice(self.n_states, p=probs)
            
            # genera azioni in base allo stato
            lam = self.mean_daily_actions[i] * self.state_multipliers[new_state[i]]
            actions[i] = np.random.poisson(lam) if lam > 0 else 0
        
        self.state = new_state
        return actions


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

    # Firehose structures
    firehose_buffer = []
    firehose_chunk = []

    # Clock for time stamp generation
    clock = ClockManager() #n_users=len(users), puda=0.036)

    # Trasforma la lista utenti in una deque per rotazione veloce
    batch_size = min(batch_size, len(users))
    users = deque(users)
    rnd.shuffle(users)

    # Experimental time handling
    is_new_day = True
    mean_daily_actions = [u.mean_action_per_day for u in users]
    mean_daily_actions = np.array(mean_daily_actions)
    sampled_users = []
    sampled_size = 0
    user_activity_sim = UserActivitySimulator(mean_daily_actions)

    # Process status
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

                        # Load the firehose chunk
                        for msg in new_msgs:
                            firehose_chunk.append(msg)

                        # Updating main structures
                        outgoing_messages[user.uid].extend(new_msgs)  # type: ignore
                        outgoing_passivities[user.uid].extend(passive_actions)  # type: ignore

                        # Updating user object
                        users_dict[user.uid] = user  # type: ignore

                    # Timestamping
                    rnd.shuffle(firehose_chunk)
                    for msg in firehose_chunk:
                        msg.time = clock.next_timestamp()

                    firehose_buffer.append(firehose_chunk)

                elif sender == "recommender_system":

                    user_pack_batch = []

                    # Count total sent users
                    # round_counter = 0

                    for _ in range(batch_size):

                        if is_new_day:
                            # 1. Pre-generate actions numbers (NOTE: change for more realism)
                            # e.g. daily_actions = np.random.poisson(mean_daily_actions) # integer vector
                            daily_actions = user_activity_sim.step()
                            # 2. Compute sampled users subset
                            active_users = [(u, daily_actions[n]) for n, u in enumerate(users) if daily_actions[n] > 0]
                            print(f"sampled {len(active_users)} active users", flush=True)
                            lurkers = [(u, 0) for n, u in enumerate(users) if daily_actions[n] == 0]
                            lurkers = rnd.sample(lurkers, k=int(len(lurkers) * 0.3)) # 30% lurkers
                            print(f"sampled {len(lurkers)} lurkers", flush=True)
                            active_users.extend(lurkers)
                            sampled_users = active_users
                            sampled_size = len(sampled_users) - 1
                            print(f"sampled {sampled_size} total users", flush=True)
                            clock.start_new_day(daily_actions)
                            print(f"total actions to be performed: {np.sum(daily_actions)}", flush=True)
                            is_new_day = False

                        # Round-robin: pick a user from tail and put in the head
                        # picked_user = users.popleft()
                        # users.append(picked_user)
                        # pick a user

                        # New version: active user subsampling
                        picked_user_index = rnd.randint(a=0, b=sampled_size)
                        picked_user, actions_today = sampled_users.pop(picked_user_index)
                        users_dict[picked_user.uid].actions_today = actions_today
                        sampled_size -= 1

                        # round_counter += 1

                        # Prepare actions to send
                        active_actions_send = outgoing_messages[picked_user.uid]
                        passive_actions_send = outgoing_passivities[picked_user.uid]

                        # Create the batch tuple
                        user_pack_batch.append(
                            (
                                users_dict[picked_user.uid],
                                active_actions_send,
                                passive_actions_send,
                            )
                        )

                        # Flush outgoing
                        outgoing_messages[picked_user.uid].clear()
                        outgoing_passivities[picked_user.uid].clear()

                        # # Shuffle users when every one has been selected
                        # if round_counter == len(users):
                        #     rnd.shuffle(users)
                        #     round_counter = 0

                        # Picked the last sampled user: start a new day
                        if sampled_size == 0:
                            print("starting new day", flush=True)
                            is_new_day = True

                    # Firehose data
                    firehose_flush = []
                    if len(firehose_buffer) > 0:
                        firehose_flush = firehose_buffer.pop(0)

                    # print(
                    #     f"[{gettimestamp()}] DataMngr >> sending {len(firehose_flush)} messages",
                    #     flush=True,
                    # )

                    # print(
                    #     f"[{gettimestamp()}] DataMngr >> firehose buffer size: {len(firehose_buffer)}",
                    #     flush=True,
                    # )

                    comm_world.send(
                        ("data_manager", (user_pack_batch, firehose_flush)),
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

    print(f"[{gettimestamp()}] DataMngr >> final clock: {clock.clock} ")
