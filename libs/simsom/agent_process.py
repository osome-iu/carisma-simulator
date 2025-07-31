"""
An agent receives inventory of messages (agent/user object) from the agent_pool_manager
and post/repost messages that will be shown to their followers
"""

from mpi4py import MPI
from mpi_utils import iprobe_with_timeout


def run_agent(
    comm_world: MPI.Intracomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    print(f"* Agent process @rank: {rank} >> running...", flush=True)

    # Status of the processes
    status = MPI.Status()

    # Process status
    alive = True

    # Process isends
    isends = []

    # Bootstrap sync
    comm_world.barrier()

    while True:

        if iprobe_with_timeout(
            comm_world,
            source=MPI.ANY_SOURCE,
            tag=MPI.ANY_TAG,
            status=status,
            timeout=5,
            pname=f"Worker_{rank}",
        ):

            # Receive package that contains (friend ids, messages) from agent_pool_manager
            sender, payload = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

            # Check if termination signal has been sent
            if sender == "analyzer" and payload == "STOP" and alive:
                print(f"* Worker_{rank} >> stop signal detected", flush=True)
                alive = False
            elif payload == "STOP" and alive:
                print(f"* Worker_{rank} >> crashing...", flush=True)
                alive = False

            # Wait for pending isends
            MPI.Request.waitall(isends)
            isends.clear()

            if alive and sender == "agntPoolMngr":

                user = payload  # Just for readability
                activities, passivities = user.make_actions()  # type: ignore

                # Repack the user (updated feed) and activities (messages he produced)
                processed_user_pack = (user, activities, passivities)

                # Send the processed user first to the data manager
                isends.append(
                    comm_world.isend(
                        ("worker", processed_user_pack),
                        dest=rank_index["data_manager"],
                    )
                )

                # Then send the processed user to the policy process
                isends.append(
                    comm_world.isend(
                        ("worker", user),
                        dest=rank_index["policy_filter"],
                    )
                )

        else:

            print(
                f"* Worker_{rank} >> closing with {len(isends)} isends...",
                flush=True,
            )

            if alive:
                print(f"* Worker_{rank} >> crashing...", flush=True)

            print(f"* Worker_{rank} >> entering barrier...", flush=True)
            comm_world.barrier()
            break

    print(f"* Worker_{rank} >> closed.", flush=True)
