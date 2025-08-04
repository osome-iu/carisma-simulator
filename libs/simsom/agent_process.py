"""
An agent receives inventory of messages (agent/user object) from the agent_pool_manager
and post/repost messages that will be shown to their followers
"""

from mpi4py import MPI
from mpi_utils import iprobe_with_timeout, gettimestamp
import os


def run_agent(
    comm_world: MPI.Intracomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    print(
        f"* Agent process @rank: {rank} (PID: {os.getpid()}) >> running...", flush=True
    )

    # Status of the processes
    status = MPI.Status()

    # Process status
    alive = True

    # Process isends
    isends = []

    dm_batch = []
    pm_batch = []

    # Bootstrap sync
    comm_world.barrier()

    while True:

        if iprobe_with_timeout(
            comm_world,
            source=MPI.ANY_SOURCE,
            tag=MPI.ANY_TAG,
            status=status,
            pname=f"Worker_{rank}",
        ):

            # Receive package that contains (friend ids, messages) from agent_pool_manager
            sender, payload = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

            # Check if termination signal has been sent
            if alive and payload == "STOP":
                print(f"* Worker_{rank} >> stop signal detected", flush=True)
                MPI.Request.waitall(isends)
                alive = False

            if alive:

                user = payload  # Just for readability
                activities, passivities = user.make_actions()  # type: ignore

                # Repack the user (updated feed) and activities (messages he produced)
                processed_user_pack = (user, activities, passivities)

                # Wait for pending isends
                if len(isends) > 0:
                    print(
                        f"* ({gettimestamp()}) Worker_{rank} >> waiting isends",
                        flush=True,
                    )
                    MPI.Request.waitall(isends)
                    isends.clear()

                dm_batch.append(processed_user_pack)
                pm_batch.append(user)

                if len(dm_batch) >= 64:

                    # Send the processed user first to the data manager
                    print(
                        f"* ({gettimestamp()}) Worker_{rank} >> sending to data_manager (rank {rank_index['data_manager']})",
                        flush=True,
                    )
                    # isends.append(
                    #     comm_world.isend(
                    #         ("worker", processed_user_pack),
                    #         dest=rank_index["data_manager"],
                    #     )
                    # )

                    isends.append(
                        comm_world.isend(
                            ("worker", dm_batch),
                            dest=rank_index["data_manager"],
                        )
                    )

                    print(
                        f"* ({gettimestamp()}) Worker_{rank} >> sending to policy_filter (rank {rank_index['policy_filter']})",
                        flush=True,
                    )

                    # # Then send the processed user to the policy process
                    # isends.append(
                    #     comm_world.isend(
                    #         ("worker", user),
                    #         dest=rank_index["policy_filter"],
                    #     )
                    # )

                    isends.append(
                        comm_world.isend(
                            ("worker", pm_batch),
                            dest=rank_index["policy_filter"],
                        )
                    )

                    dm_batch.clear()
                    pm_batch.clear()

        else:

            print(
                f"* Worker_{rank} >> closing with {len(isends)} isends...",
                flush=True,
            )

            if alive:

                print(f"* Worker_{rank} >> crashing...", flush=True)
                MPI.Request.waitall(isends)
                # TODO: handle crash

            print(f"* Worker_{rank} >> entering barrier...", flush=True)
            comm_world.barrier()
            break

    print(f"* Worker_{rank} >> closed.", flush=True)
