from mpi4py import MPI
from mpi_utils import iprobe_with_timeout, handle_crash


def run_policy_filter(
    comm_world: MPI.Intracomm,
    rank: int,
    size: int,  # If needed for future logic
    rank_index: dict,
):

    print("* Policy process >> running...", flush=True)

    # Status of the processes
    status = MPI.Status()

    # DEBUG COUNTER
    count = 0

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
            pname="PolicyMngr",
        ):

            sender, payload = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

            # Check if termination signal has been sent
            if alive and payload == "STOP":
                print("* PolicyMngr >> stop signal detected...", flush=True)
                MPI.Request.waitall(isends)
                isends.clear()
                alive = False

            if alive:

                _ = payload  # TO BE IMPLEMENTED

                count += 1

                # Wait for pending isends
                if len(isends) > 100:
                    MPI.Request.waitall(isends)
                    isends.clear()

                if count == 10:

                    isends.append(
                        comm_world.isend(
                            ("policyMngr", None),
                            dest=rank_index["data_manager"],
                        )
                    )

        else:

            print(f"* PolicyMngr >> closing with {len(isends)} isends...", flush=True)

            if alive:

                MPI.Request.waitall(isends)

                handle_crash(
                    comm_world=comm_world,
                    status=status,
                    srank=rank,
                    srole="policy_filter",
                    pname="PolicyMngr",
                )

            print("* PolicyMngr >> entering barrier...", flush=True)
            comm_world.barrier()
            break

    print("* PolicyMngr >> closed.", flush=True)
