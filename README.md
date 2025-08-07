# SimSoM: A <ins>Sim</ins>ulator of <ins>So</ins>cial <ins>M</ins>edia

This repository contains the latest version of the simulator. This version uses a Message Passing Interface [mpi4py](https://mpi4py.readthedocs.io/en/stable/overview.html) to run the simulator on parallel computers. 

## Overview

The definitions of the `Agent` (`User` in the future) and `Message` classes are kept to a minimum to facilitate code testing. However, please note that many features are still missing, such as:

- Messages produced are not saved to disk
- There is no convergence checking
- Logger removed due to more than one process
- No timestamp clock

The current code focuses on implementing the agent pool manager and the mechanism for scheduling agents to each agent handler process when available.

## Batched Agent Processing

This implementation processes agents in batches. The batch size can be configured to optimize performance based on the specific hardware and workload.

## Architecture

The logical target architecture of the system is illustrated in the following diagram:

![Logical Architecture](architecture/simsom_2025-08-05.jpg)

## Planned Improvements

The current implementation has very basic definitions for the `Message` and `Agent` classes. However, the previous development version included more complex versions of these classes that need to be ported over.

- Porting of previous `Message` and `Agent` (`User`) classes
- Add configuration settings to command line (now are hardcoded)
- Convergence process logic implementation
- Persistent storage of generated messages to disk
- Timestamp-based message ordering and clock management

## Usage

To start the simulation, use the following command:

```
mpiexec -n 10 python simsom.py --network_spec "./config/default_network_config.json" --simulator_spec "./config/default_simulator_config.json"
```

This will run the simulation using 10 processes.

## Known Limitations

- **Missing Features**: As mentioned in the docstring, many features are still missing, such as message persistence, convergence checking, and timestamp clock.
- **Experimental Code**: This implementation is still in an experimental stage and should be used with caution.

## Contact

For any questions please contact the following people:

- Bao Tran Truong: `baotruon@iu.edu`
- Gianluca Nogara: `gianluca.nogara@supsi.ch`
- Enrico Verdolotti: `enrico.verdolotti@supsi.ch`
