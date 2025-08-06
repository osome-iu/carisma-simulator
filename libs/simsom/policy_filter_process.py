from mpi4py import MPI
from mpi_utils import iprobe_with_timeout, handle_crash, gettimestamp
import os


def run_policy_filter(
    comm_world: MPI.Intracomm,
    rank: int,
    size: int,  # If needed for future logic
    rank_index: dict,
):

    print(
        f"[{gettimestamp()}] PolicyEval (PID: {os.getpid()}) >> running...", flush=True
    )

    # Status of the processes
    status = MPI.Status()

    # DEBUG COUNTER
    count = 0

    # Process status
    alive = True

    # Bootstrap sync
    comm_world.barrier()

    while True:

        if iprobe_with_timeout(comm_world, status=status, pname="PolicyMngr"):

            sender, payload = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

            # Check if termination signal has been sent
            if alive and payload == "STOP":
                print(
                    f"[{gettimestamp()}] PolicyEval >> stop signal detected...",
                    flush=True,
                )

                alive = False

            if alive:

                # print(
                #     f"[{gettimestamp()}] PolicyEval > processed user received from {sender}!",
                #     flush=True,
                # )

                _ = payload  # TO BE IMPLEMENTED

                count += 1

                if count == 10:

                    comm_world.send(
                        ("policy_evaluator", None),
                        dest=rank_index["data_manager"],
                    )

        else:

            print(f"[{gettimestamp()}] PolicyEval >> closing...", flush=True)

            if alive:

                handle_crash(
                    comm_world=comm_world,
                    status=status,
                    srank=rank,
                    srole="policy_evaluator",
                    pname="PolicyMngr (crashed)",
                )

            print(f"[{gettimestamp()}] PolicyEval >> entering barrier...", flush=True)
            comm_world.barrier()
            break

    print(f"[{gettimestamp()}] PolicyEval >> closed.", flush=True)
