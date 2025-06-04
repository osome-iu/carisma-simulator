from mpi4py import MPI
import time
from user import User

def suspension_base(user, users_packs_batch, current_time):
    """
    Handles user suspension and removes their messages from newsfeeds of other users.

    Args:
        user (User): The user being processed.
        users_packs_batch (list): List of (user, in_messages, current_time) tuples.
        current_time (float): The current simulation time.
    """
    # Skip terminated users immediately
    if user.is_terminated:
        return

    # Constants
    STRIKE_WINDOW = 9     # Time window to evaluate strikes (e.g., 9 days for now)
    SUSPENSION_DURATIONS = {1: 1, 2: 2}  # Suspension lengths per strike count (in days -- 1 and 2 for now)

    # Remove expired strikes (outside the 90-day window)
    user.strike_timestamps = [
        ts for ts in user.strike_timestamps if current_time - ts <= STRIKE_WINDOW
    ]

    # Check if suspension should be lifted
    if user.is_suspended and current_time >= user.suspension_lift_time:
        user.is_suspended = False

    # Handle bad message posting (new strike)
    if user.bad_message_posting:
        user.bad_message_posting = False
        user.strike_timestamps.append(current_time)
        user.sus_strike_count = len(user.strike_timestamps)

        # Immediate termination if 3+ strikes within STRIKE_WINDOW
        if user.sus_strike_count >= 3:
            user.is_terminated = True
            return

        # Suspend user for appropriate duration
        user.is_suspended = True
        user.suspended_time = current_time
        suspension_duration = SUSPENSION_DURATIONS.get(user.sus_strike_count, 14)
        user.suspension_lift_time = current_time + suspension_duration

        # Clear user's own newsfeed if it exists
        if hasattr(user, "newsfeed"):
            user.newsfeed = []

        # Remove user's messages from others' newsfeeds
        for other_user, _, _ in users_packs_batch:
            if hasattr(other_user, "newsfeed"):
                other_user.newsfeed = [
                    msg for msg in other_user.newsfeed if msg.uid != user.uid
                ]

def run_policy_filter(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,  # If needed for future logic
    rank_index: dict,
):

    # Verbose: use flush=True to print messages
    print("- Policy process >> started", flush=True)

    # Status of the processes
    status = MPI.Status()

    # DEBUG COUNTER
    count = 0

    # Bootstrap sync
    comm_world.Barrier()

    while True:

        data = comm_world.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        if data == "sigterm":
            # print("- Policy filter >> termination signal")

            # Flush pending incoming messages
            while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
                _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
            comm_world.Barrier()
            break
        # Now we know `data` is safe to process (not a string)
        user_packs_batch = data

        # Defensive structure check #SY HERE
        if isinstance(user_packs_batch, User):
            raise TypeError("Expected a batch of (user, in_messages, current_time), got a single User object.")
        elif not isinstance(user_packs_batch, list):
            user_packs_batch = [user_packs_batch]

        for i, user_pack in enumerate(user_packs_batch):
            user, in_messages, current_time = user_pack
            suspension_base(user, user_packs_batch, current_time)
            user_packs_batch[i] = (user, in_messages, current_time)

        comm_world.send(user_packs_batch, dest=rank_index["data_manager"])
        
        count += 1

        if count == 10:
            comm_world.send(("ping_policy", 0), dest=rank_index["data_manager"])
