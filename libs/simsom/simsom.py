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

import sys
import json
from mpi4py import MPI
import simtools

from data_manager_process import run_data_manager
from convergence_monitor_process import run_convergence_monitor
from policy_filter_process import run_policy_filter
from agent_pool_manager_process import run_agent_pool_manager
from agent_process import run_agent


# Configuration constants
RANK_INDEX = {
    "data_manager": 0,
    "convergence_monitor": 1,
    "policy_filter": 2,
    "agent_pool_manager": 3,
    "agent_handler": 4,
}

with open("config.json", "r", encoding="utf-8") as file:
    config = json.load(file)


def main():

    comm_world = MPI.COMM_WORLD
    size = comm_world.Get_size()
    rank = comm_world.Get_rank()

    # Simulation contstraints (parametrize)
    users = (
        simtools.init_network(file=config["real_world_netowork"])
        if config["real_world_netowork"]
        else simtools.init_network(
            net_size=config["net_size"],
            p=config["probability_follow"],
            k_out=config["avg_n_friend"],
        )
    )

    if size < 5:
        if rank == 0:
            print("Error: This program requires at least 5 processes")
        sys.exit(1)

    if rank == RANK_INDEX["data_manager"]:
        run_data_manager(
            users=users,
            message_count_target=config["message_count_target"],
            comm_world=comm_world,
            rank=rank,
            size=size,
            rank_index=RANK_INDEX,
        )

    elif rank == RANK_INDEX["convergence_monitor"]:
        run_convergence_monitor(
            comm_world=comm_world,
            rank=rank,
            rank_index=RANK_INDEX,
            sliding_window_convergence=config["sliding_window_convergence"],
            message_count_target=config["message_count_target"],
            convergence_param=config["threshold_convergence"],
        )

    elif rank == RANK_INDEX["policy_filter"]:
        run_policy_filter(
            comm_world=comm_world, rank=rank, size=size, rank_index=RANK_INDEX
        )

    elif rank == RANK_INDEX["agent_pool_manager"]:
        run_agent_pool_manager(
            comm_world=comm_world, rank=rank, size=size, rank_index=RANK_INDEX
        )

    elif rank >= RANK_INDEX["agent_handler"]:
        run_agent(comm_world=comm_world, rank=rank, size=size, rank_index=RANK_INDEX)


if __name__ == "__main__":
    main()
