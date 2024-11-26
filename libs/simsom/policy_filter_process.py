from mpi4py import MPI


def run_policy_filter(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,  # If needed for future logic
    rank_index: dict,
):

    print("Policy filter start")

    # Status of the processes
    status = MPI.Status()

    # Bootstrap sync
    comm_world.Barrier()

    while True:

        # Wait for a batch of (agents, in_messages) to process
        agent_packs_batch = comm_world.recv(
            source=rank_index["data_manager"], status=status
        )

        processed_batch = agent_packs_batch  # Placeholder for future logic

        # Redirect the processed batch to agent pool manager
        comm_world.send(processed_batch, dest=rank_index["agent_pool_manager"])

        # Check for termination signal
        if agent_packs_batch == "sigterm":
            break

    print(f"Policy filter stop @ rank: {rank}")
