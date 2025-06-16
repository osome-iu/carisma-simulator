from mpi4py import MPI
import time


def safe_finalize_isends(requests, soft_checks=3, hard_timeout=0.1):
    pending = requests.copy()
    for _ in range(soft_checks):
        if MPI.Request.Testall(pending):
            return
    # fallback with sleep
    start = time.time()
    while time.time() - start < hard_timeout and pending:
        pending = [req for req in pending if not req.Test()]


def flush_incoming_messages(comm, status):
    while comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
        _ = comm.recv(source=status.Get_source(), tag=status.Get_tag(), status=status)


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

    # Bootstrap sync
    comm_world.barrier()

    while True:

        data = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

        # Check for termination
        if status.Get_tag() == 99:
            print("PolicyProc >> sigterm detected", flush=True)
            flush_incoming_messages(comm_world, status)
            comm_world.barrier()
            break

        _ = data  # TO BE IMPLEMENTED

        # if data == "sigterm":
        #     # print("- Policy filter >> termination signal")

        #     # Flush pending incoming messages
        #     while comm_world.Iprobe(source=MPI.ANY_SOURCE, status=status):
        #         _ = comm_world.recv(source=MPI.ANY_SOURCE, status=status)
        #     comm_world.Barrier()
        #     break

        count += 1

        if count == 10:

            req1 = comm_world.isend(
                ("policy_proc", None), dest=rank_index["data_manager"]
            )

            safe_finalize_isends([req1])

    print("* PolicyProc >> closed.", flush=True)
