"""
This module defines a basic schema for the parallel implementation of SimSoM originally
developed by Bao Tran Truong from Indiana University.

The definitions of the Agent and Message classes are kept to a minimum
to facilitate code testing.

WARNING: Many features are still missing, e.g., messages produced are not saved to disk,
there is no convergence checking, no timestamp clock, etc.
The current code focuses on implementing the agent pool manager and
the mechanism for scheduling agents to each agent handler process when available.

Example of starting command: mpiexec -n 8 python simsom.py

For any questions, please contact us at:

- baotruon@iu.edu
- gianluca.nogara@supsi.ch
- enrico.verdolotti@supsi.ch

"""

from mpi4py import MPI
import sys


# Configuration constants
RANK_INDEX = {
    "data_manager": 0,
    "convergence_monitor": 1,
    "policy_filter": 2,
    "agent_pool_manager": 3,
    "agent_handler": 4,
}


def main():

    comm_world = MPI.COMM_WORLD
    size = comm_world.Get_size()
    rank = comm_world.Get_rank()

    if size < 5:
        if rank == 0:
            print("Error: This program requires at least 5 processes")
        sys.exit(1)

    if rank == RANK_INDEX["data_manager"]:

        from data_manager_process import run_data_manager

        run_data_manager(comm_world, rank, size, RANK_INDEX)

    elif rank == RANK_INDEX["convergence_monitor"]:

        from convergence_monitor_process import run_convergence_monitor

        run_convergence_monitor(comm_world, rank, size, RANK_INDEX)

    elif rank == RANK_INDEX["policy_filter"]:

        from policy_filter_process import run_policy_filter

        run_policy_filter(comm_world, rank, size, RANK_INDEX)

    elif rank == RANK_INDEX["agent_pool_manager"]:

        from agent_pool_manager_process import run_agent_pool_manager

        run_agent_pool_manager(comm_world, rank, size, RANK_INDEX)

    elif rank >= RANK_INDEX["agent_handler"]:

        from agent_process import run_agent

        run_agent(comm_world, rank, size, RANK_INDEX)


if __name__ == "__main__":
    main()
