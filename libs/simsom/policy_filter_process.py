from mpi4py import MPI
import time

def suspension_base(user, current_time):
    """
    Handles user suspension and removes their messages from newsfeeds of other users.

    Args:
        user (User): The user being processed.
        current_time (float): The current simulation time.
    """
    # Skip terminated users immediately
    if user.is_terminated:
        return

    # Constants
    STRIKE_WINDOW = 0.1     # Time window to evaluate strikes (e.g., 9 days for now)

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
        suspension_duration = 0.0002 * user.sus_strike_count
        user.suspension_lift_time = current_time + suspension_duration

        # Clear user's own newsfeed
        user.newsfeed = []

def suspension_abrupt(user, current_time):
    """
    Handles user suspension and removes their messages from newsfeeds of other users.
    This version is used for abrupt change of user behaviors, where the user is stop posting bad messages after the first strike.

    Args:
        user (User): The user being processed.
        current_time (float): The current simulation time.
    """
    # Skip if already suspended and check for lifting suspension
    if user.is_suspended and current_time >= user.suspension_lift_time:
        user.is_suspended = False

    # Skip if user is already marked as permanently reformed
    if user.no_bad_posting:
        return

    STRIKE_WINDOW = 0.1  # e.g., ~9 days

    # Remove expired strikes
    user.strike_timestamps = [
        ts for ts in user.strike_timestamps if current_time - ts <= STRIKE_WINDOW
    ]

    # If the user posted a bad message this round
    if user.bad_message_posting:
        user.bad_message_posting = False  # Reset this round's flag
        user.strike_timestamps.append(current_time)
        user.sus_strike_count = len(user.strike_timestamps)

        # User changes behavior after first strike
        user.no_bad_posting = True

        # Suspend the user
        user.is_suspended = True
        user.suspended_time = current_time
        suspension_duration = 0.0002 * user.sus_strike_count
        user.suspension_lift_time = current_time + suspension_duration

        # Clear their own newsfeed
        user.newsfeed = []

        # Optional log
        # print(f"[INFO] User {user.uid} suspended and reformed after first strike.")

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
            print("- Policy filter >> termination signal")

            # Flush pending incoming messages
            while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
                _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
            comm_world.Barrier()
            break

        user, current_time = data
        suspension_base(user, current_time) #apply suspension logic

        # Send back the updated user
        #print("- Policy filter >> data manager")
        comm_world.send(("ping_policy", (user, current_time)), dest=rank_index["data_manager"])

        #count += 1 <-- Guess this is for testing purpose?

        #if count == 10:
        #    comm_world.send(("ping_policy", 0), dest=rank_index["data_manager"])
