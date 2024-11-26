"""
Reponsible for monitoring the convergence of the simulation.
Send termination signal to all processes when the simulation has converged.
"""
from mpi4py import MPI


def run_convergence_monitor(
    comm_world: MPI.Intercomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    print("Convergence monitor start")

    # Status of the processes
    # status = MPI.Status()

    # Bootstrap sync
    comm_world.Barrier()

    # TODO: Implement future convergence criteria for termination.
    # The main concept is that this process read directly from disk.
    # This should be the only other process that can send
    # a termination signal to all the other processes.

    print(f"Convergence monitor stop @ rank: {rank}")
