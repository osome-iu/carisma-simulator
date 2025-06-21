from mpi4py import MPI
from mpi_utils import iprobe_with_timeout


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
        ):

            data = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

            # Check for termination
            if status.Get_tag() == 99:
                print("* PolicyProc >> stop signal detected", flush=True)
                alive = False

            MPI.Request.waitall(isends)
            isends.clear()

            if alive:

                _ = data  # TO BE IMPLEMENTED

                count += 1

                if count == 10:

                    req1 = comm_world.isend(
                        ("policy_proc", None),
                        dest=rank_index["data_manager"],
                    )

                    isends.append(req1)

        else:
            print("* PolicyProc >> waiting isends...", flush=True)
            MPI.Request.waitall(isends)
            print("* PolicyProc >> entering barrier...", flush=True)
            comm_world.barrier()
            break

    print("* PolicyProc >> closed.", flush=True)
