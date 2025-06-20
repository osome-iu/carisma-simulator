"""
An agent receives inventory of messages (agent/user object) from the agent_pool_manager
and post/repost messages that will be shown to their followers
"""

from mpi4py import MPI
from mpi_utils import iprobe_with_timeout


def run_agent(
    comm_world: MPI.Intracomm,
    rank: int,
    size: int,
    rank_index: dict,
):

    print(f"* Agent process @rank: {rank} >> running...", flush=True)

    # Status of the processes
    status = MPI.Status()

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

            # Receive package that contains (friend ids, messages) from agent_pool_manager
            data = comm_world.recv(source=MPI.ANY_SOURCE, status=status)

            # Check for termination
            if status.Get_tag() == 99:
                print(f"Agent@{rank} >> stop signal detected", flush=True)
                alive = False

            MPI.Request.waitall(isends)
            isends.clear()

            if alive:

                user = data  # Just for readability
                activities, passivities = user.make_actions()

                # Repack the agent (updated feed) and activities (messages he produced)
                agnt_pack_rep = ("agent_proc", (user, activities, passivities))

                # Send the processed user first to the data manager
                req1 = comm_world.isend(agnt_pack_rep, dest=rank_index["data_manager"])

                # Then send the processed user to the policy process
                req2 = comm_world.isend(user, dest=rank_index["policy_filter"])

                isends.append(req1)
                isends.append(req2)

            else:

                print(f"* Agent@{rank} >> Not sending stuff.", flush=True)

        else:

            print(f"Agent@{rank} >> Entering barrier...", flush=True)
            comm_world.barrier()
            print(f"Agent@{rank} >> Barrier passed.", flush=True)
            break

    print(f"* Agent process @rank: {rank} >> closed.", flush=True)
