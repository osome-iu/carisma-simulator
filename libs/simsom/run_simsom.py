"""
Utiliy script to run the simulation with 'python run_simsom.py'
"""

import subprocess
import argparse


def main():

    # Default simulation parameters (change these manually if needed)
    default_num_procs = 32

    # Optional CLI override
    parser = argparse.ArgumentParser(description="Run the MPI simulation.")

    # Num workers param
    parser.add_argument(
        "-n",
        "--num-procs",
        type=int,
        default=default_num_procs,
        help="Number of MPI processes",
    )

    # Script name
    parser.add_argument(
        "--script",
        type=str,
        default="simsom.py",
        help="Simulation entry script",
    )

    args = parser.parse_args()

    # Format arguments for subprocess
    command = ["mpiexec", "-n", str(args.num_procs), "python", args.script]

    print("Running command:", " ".join(command))
    subprocess.run(command)


if __name__ == "__main__":
    main()
