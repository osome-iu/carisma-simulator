from mpi4py import MPI
import time

def suspension(user, users_packs_batch, current_time):
    """
    Handles user suspension and removes their messages from others' newsfeeds.

    Args:
        user (User): The user being processed.
        user_packs_batch (list): List of (user, in_messages, current_time) tuples.
        current_time (float): The current simulation time.
    """
    # Skip terminated users immediately
    if user.is_terminated:
        return
    
    # Check if suspension time has passed
    if user.is_suspended and abs(current_time - user.suspended_time) > 1:
        user.is_suspended = False  # Lift suspension

    # Check if flagging time has passed
    if user.is_flagged and abs(current_time - user.current_flagged_time) > 6:
        if not user.is_second_flagged:
            user.is_flagged = False
            user.is_first_flagged = False
            user.current_flagged_time = 0
            user.first_flagged_time = 0
        if user.is_second_flagged: #make a second flag as a first flag
            user.is_first_flagged = True
            user.first_flagged_time = user.second_flagged_time
            user.current_flagged_time = user.second_flagged_time
            user.is_second_flagged = False
            user.second_flagged_time = 0

    # If the user posted a bad message
    if user.bad_message_posting:
        if not user.is_flagged:
            # First offense or flag reset: Flag and suspend
            user.is_flagged = True
            user.is_first_flagged = True
            user.is_suspended = True
            user.sus_strike_count += 1
            user.first_flagged_time = current_time
            user.current_flagged_time = current_time
            user.suspended_time = current_time
        else:
            # Already flagged → Just suspend, increase strike count
            user.is_suspended = True
            user.is_second_flagged = True
            user.second_flagged_time = current_time
            user.suspended_time = current_time
            user.sus_strike_count += 1

        # **Clear the suspended user's feed**
        if hasattr(user, "newsfeed"):
            user.newsfeed = []

        # **Remove suspended user’s messages from other users' newsfeeds**
        for other_user, _, _ in users_packs_batch:
            if hasattr(other_user, "newsfeed"):
                other_user.newsfeed = [
                    msg for msg in other_user.newsfeed if msg.uid != user.uid
                ]

    # Check for account termination
    if user.sus_strike_count >= 3:
        user.is_terminated = True

def run_policy_filter(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,  # If needed for future logic
    rank_index: dict,
):

    # Status of the processes
    status = MPI.Status()

    # Bootstrap sync
    comm_world.Barrier()

    while True:

        # Wait for a batch of (agents, in_messages) to process
        print(f"Policy filter @ rank {rank} waiting for batch from data manager", flush=True)
        user_packs_batch = comm_world.recv(
            source=rank_index["data_manager"], status=status
        )
        print(f"Policy filter @ rank {rank} received batch", flush=True)

        # Check for termination signal
        if user_packs_batch == "sigterm":
            comm_world.send("sigterm", dest=rank_index["agent_pool_manager"])
            break
        # Process each user pack

        for i, user_pack in enumerate(user_packs_batch):
            user, in_messages, current_time = user_pack

            # Apply suspension logic using the batch itself
            suspension(user, user_packs_batch, current_time)
            print(f"User {user.uid} went through susension: suspension - {user.is_suspended}, termination - {user.is_terminated}", flush=True)
            # Update the user pack
            user_packs_batch[i] = (user, in_messages, current_time)

        processed_batch = user_packs_batch

        # Redirect the processed batch to agent pool manager
        #print(f"Sending batch from policy filter @ rank {rank}", flush=True)
        comm_world.send(processed_batch, dest=rank_index["agent_pool_manager"])
