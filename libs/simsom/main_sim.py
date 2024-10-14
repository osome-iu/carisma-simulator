"""
Put a module docstring here :)
# hello_mpi.py:
# usage: mpiexec -n 10 python main_sim.py --n_user 50 --stop_iteration 100
"""

from mpi4py import MPI
import argparse
import time

# Projet import
from simtools import init_network, file_manager

# Do we need this?
PROCESSES = {
    0: "convergence_manager",
    1: "simulation_master",
    2: "message_buffer",
    3: "data_manager",
}


TERMINATION_SIGNAL = "STOP"
FILE_PATH = f"files/file_{int(time.time())}.csv"
FOLDER_PATH = "files"


def run_simulation():
    """
    Simulation driver function
    """

    # MPI Context init
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    comm.Set_errhandler(MPI.ERRORS_RETURN)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_user", type=int, default=20, help="Number of users simulated"
    )

    parser.add_argument(
        "--stop_iteration",
        type=int,
        default=0,
        help="Number of iterations to be reached to stop the execution (Default 0, run with convergence method)",
    )

    args = parser.parse_args()

    # Simulation contstraints (parametrize)
    no_processes = size
    users = init_network(args.n_user)

    if no_processes < 4:
        if rank == 0:
            print("Please run again with at least 5 processes")
    else:
        if rank == 0:
            from convergence_manager import convergence

            file_manager(FOLDER_PATH, FILE_PATH)
            convergence(comm, rank, 100, FILE_PATH, args.stop_iteration, 0.01)

        if rank == 1:
            from simulation_master_process import simulation_master

            simulation_master(
                comm,
                rank,
                size,
                users,
                args.stop_iteration,
                sigterm=TERMINATION_SIGNAL,
            )

        if rank == 2:
            from message_buffer_process import message_buffer

            message_buffer(
                comm,
                rank,
                args.stop_iteration,
                sigterm=TERMINATION_SIGNAL,
            )

        if rank == 3:
            from data_manager_process import data_manager

            data_manager(
                comm,
                rank,
                file_path=FILE_PATH,
                sigterm=TERMINATION_SIGNAL,
            )

        if rank >= 4:
            from agent_handler_process import agent_handler

            agent_handler(
                comm,
                rank,
                sigterm=TERMINATION_SIGNAL,
            )


if __name__ == "__main__":

    run_simulation()
