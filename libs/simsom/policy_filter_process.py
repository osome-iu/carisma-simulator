from mpi4py import MPI


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

        count += 1

        if count == 10:
            comm_world.send(("ping_policy", 0), dest=rank_index["data_manager"])

    print("* Policy process >> closed.", flush=True)
