"""
An agent receives inventory of messages (agent/user object) from the agent_pool_manager
and post/repost messages that will be shown to their followers
"""

from mpi4py import MPI
from mpi_utils import iprobe_with_timeout, gettimestamp
import time
import os


def run_agent(
    comm_world: MPI.Intracomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    print(
        f"[{gettimestamp()}] Worker_{rank} (PID: {os.getpid()}) > running...",
        flush=True,
    )

    # Status of the processes
    status = MPI.Status()

    # Process status
    alive = True

    out_batch = []

    # Bootstrap sync
    comm_world.barrier()

    while True:

        if alive:

            # print(f"[{gettimestamp()}] Worker_{rank} > requesting data...", flush=True)
            comm_world.send(("worker", rank), dest=rank_index["recommender_system"])

        if iprobe_with_timeout(comm_world, status=status, pname=f"Worker_{rank}"):

            # print(f"[{gettimestamp()}] Worker_{rank} > receiving data...", flush=True)
            sender, payload = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
            # print(
            #     f"[{gettimestamp()}] Worker_{rank} > received data from {sender}!",
            #     flush=True,
            # )

            # Check termination signal
            if alive and payload == "STOP":

                print(
                    f"[{gettimestamp()}] Worker_{rank} > stop signal detected from {sender}!",
                    flush=True,
                )

                alive = False

            if alive:

                # if payload == "wait":
                #     time.sleep(1)
                #     continue

                user = payload  # Just for readability
                activities, passivities = user.make_actions()  # type: ignore

                # Repack the data
                processed_user_pack = (user, activities, passivities)

                out_batch.append(processed_user_pack)

                if len(out_batch) >= 32:

                    # print(
                    #     f"[{gettimestamp()}] Worker_{rank} > sending to data_manager...",
                    #     flush=True,
                    # )

                    comm_world.send(
                        ("worker", out_batch),
                        dest=rank_index["data_manager"],
                    )

                    # print(
                    #     f"[{gettimestamp()}] Worker_{rank}> sending to policy_evaluator...",
                    #     flush=True,
                    # )

                    comm_world.send(
                        ("worker", out_batch),
                        dest=rank_index["policy_evaluator"],
                    )

                    out_batch.clear()

        else:

            print(
                f"[{gettimestamp()}] Worker_{rank} > closing...",
                flush=True,
            )

            if alive:

                print(f"[{gettimestamp()}] Worker_{rank} > crashing...", flush=True)
                # TODO: handle crash

            print(f"[{gettimestamp()}] Worker_{rank} > entering barrier...", flush=True)
            comm_world.barrier()
            break

    print(f"[{gettimestamp()}] Worker_{rank} > closed.", flush=True)
