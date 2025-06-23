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
import argparse

from data_manager_process import run_data_manager
from analyzer_process import run_analyzer
from policy_filter_process import run_policy_filter
from agent_pool_manager_process import run_agent_pool_manager
from agent_process import run_agent
from recommender_system import run_recommender_system


# Configuration constants
RANK_INDEX = {
    "data_manager": 0,
    "recommender_system": 1,
    "analyzer": 2,
    "agent_pool_manager": 3,
    "policy_filter": 4,
    "agent_handler": 5,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--network_spec",
    type=str,
    default="config/default_network_config.json",
    help="File that contains configuration for the network",
)
parser.add_argument(
    "--simulator_spec",
    type=str,
    default="config/default_simulator_config.json",
    help="File that contains configuration for the simulation",
)

args = parser.parse_args()

with open(args.network_spec, "r", encoding="utf-8") as file:
    network_config = json.load(file)

with open(args.simulator_spec, "r", encoding="utf-8") as file:
    simulator_config = json.load(file)


def main():

    comm_world = MPI.COMM_WORLD
    size = comm_world.Get_size()
    rank = comm_world.Get_rank()

    # Simulation contstraints (parametrize)
    users = (
        simtools.init_network(file=network_config["real_world_network"])
        if network_config["from_file"] > 0
        else simtools.init_network(
            net_size=network_config["net_size"],
            p=network_config["probability_follow"],
            k_out=network_config["avg_n_friend"],
        )
    )

    if size < 6:

        if rank == 0:
            print("Error: This program requires at least 6 processes")
        sys.exit(1)

    if rank == RANK_INDEX["data_manager"]:

        run_data_manager(
            users=users,
            comm_world=comm_world,
            rank=rank,
            size=size,
            rank_index=RANK_INDEX,
            batch_size=simulator_config["data_manager_batchsize"],
        )

    elif rank == RANK_INDEX["policy_filter"]:

        run_policy_filter(
            comm_world=comm_world,
            rank=rank,
            size=size,
            rank_index=RANK_INDEX,
        )

    elif rank == RANK_INDEX["recommender_system"]:

        run_recommender_system(
            comm_world=comm_world,
            rank=rank,
            size=size,
            rank_index=RANK_INDEX,
        )

    elif rank == RANK_INDEX["analyzer"]:

        run_analyzer(
            comm_world=comm_world,
            rank=rank,
            rank_index=RANK_INDEX,
            # Params for sliding window method
            sliding_window_method=simulator_config["sliding_window_method"],
            sliding_window_size=simulator_config["sliding_window_size"],
            sliding_window_threshold=simulator_config["sliding_window_threshold"],
            # Params for max target method
            max_interactions_method=simulator_config["max_interactions_method"],
            max_iteration_target=simulator_config["max_iteration_target"],
            # Params for historical quality
            ema_quality_method=simulator_config["ema_quality_method"],
            ema_quality_convergence=simulator_config["ema_quality_convergence"],
            # Number of users
            n_users=len(users),
            # Params for printing stuff during the execution
            verbose=simulator_config["verbose"],
            print_interval=simulator_config["print_interval"],
            # Params for saving activities on disk
            save_active_interactions=simulator_config["save_active_interactions"],
            save_passive_interactions=simulator_config["save_passive_interactions"],
        )

    elif rank == RANK_INDEX["agent_pool_manager"]:

        run_agent_pool_manager(
            comm_world=comm_world,
            rank=rank,
            size=size,
            rank_index=RANK_INDEX,
        )

    elif rank >= RANK_INDEX["agent_handler"]:
        run_agent(
            comm_world=comm_world,
            rank=rank,
            size=size,
            rank_index=RANK_INDEX,
        )


if __name__ == "__main__":

    main()
