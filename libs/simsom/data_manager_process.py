"""
The data manager is responsible for choosing Users to run, save on disk generated data and
"""

import time
import csv
import random as rnd
import copy
import numpy as np
import pandas as pd
from mpi4py import MPI
from user import User

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


def run_data_manager(
    users: list,
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,
    rank_index: dict,
    batch_size=5,
):

    # Verbose: use flush=True to print messages
    # print("- Data manager >> started", flush=True)

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
    comm_world.Barrier()
    
    print("Simulation started", flush=True)
    

    while True:

        data = comm_world.recv(source=MPI.ANY_SOURCE, status=status)        
        msg, content = data

        if msg == "ping_agent_pool_manager":
            # Unpack the agent + incoming messages and passive actions
            user, new_msgs, passive_actions = content
            for msg in new_msgs:
                msg.time = clock.next_time()   
            # print(f"- Data manager >> {user.uid} has {len(new_msgs)} new messages", flush=True)
            # print(f"- Data manager >> {user.uid} has {len(passive_actions)} new passivities", flush=True)
            
            # TODO: FIX THIS DEADPOINT, if we uncomment this we have a deadpoint
            outgoing_messages[user.uid].extend(new_msgs)
            outgoing_passivities[user.uid].extend(passive_actions)
            # print(len(outgoing_passivities[user.uid]))

        elif msg == "ping_recsys":
            # Unpicked agents count
            
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

                # Move picked user to the end of the list
                users = users[1:] + [picked_user]
                
                # Get the in and out messages based on friends
                active_actions_send = outgoing_messages[picked_user.uid]
                passive_actions_send = outgoing_passivities[picked_user.uid]

                # Add it to the batch
                current_time = clock.next_time() #get current_time as it is needed for policy
                users_packs_batch.append((picked_user, active_actions_send, passive_actions_send, current_time)) #pack current_time with others

                # TODO: Flush outgoing messages ????
                outgoing_messages[picked_user.uid] = []
                outgoing_passivities[picked_user.uid] = []

                # Before we process all the users, we need to shuffle them
                if len(selected_users) == len(users):
                    rnd.shuffle(users)
                    selected_users.clear()
                
            comm_world.send(users_packs_batch, dest=rank_index["recommender_system"])

        elif msg == "ping_policy":
            user, current_time = content
            # Find and replace the user in the current users list
            for i, existing_user in enumerate(users):
                if existing_user.uid == user.uid:
                    users[i] = user
                    break
            # print("- Data manager >> ping policy")

        elif msg == "sigterm":
            # print("- Data manager >> termination signal, stopping simulation...")

            # Flush pending incoming messages
            while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
                _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
            comm_world.Barrier()
            break
    print("- Data manager >> finished", flush=True)