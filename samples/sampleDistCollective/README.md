# TensorRT Distributed Collective Sample


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
    * [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
    * [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, `sampleDistCollective`, demonstrates how to use TensorRT for multi-GPU inference by creating and running TensorRT networks with `IDistCollectiveLayer`. It tests a specific collective operation specified via the required `--op` argument by building a network for that operation and verifying the results.

## How does this sample work?

The sample builds a TensorRT network containing a distributed collective layer, then runs inference across multiple GPUs using NCCL for GPU-to-GPU communication.

Specifically:
-   `INetworkDefinition::addDistCollective` is called to add the collective layer (kALL_REDUCE, kALL_GATHER, kBROADCAST, kREDUCE, or kREDUCE_SCATTER).
-   `IDistCollectiveLayer::setNbRanks` is called to set the number of ranks for the collective operation.
-   The NCCL unique ID is coordinated via a shared file specified by `TRT_NCCL_ID_FILE`. Rank 0 generates the ID and writes it to the file; other ranks wait and read it.
-   `ncclCommInitRank` is called to initialize the NCCL communicator on each rank.
-   `IExecutionContext::setCommunicator` is called to set the NCCL communicator on the execution context.
-   After inference, each rank verifies its output matches the expected result for the collective operation.


### TensorRT API layers and ops

In this sample, the [IDistCollectiveLayer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) is used for distributed collective operations across multiple GPUs. For more information, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

## Prerequisites

1. **Multiple GPUs**: This sample requires at least 2 GPUs.

2. **NCCL**: Install NCCL library (version should be >= 2.19.0 and < 3.0):
    ```bash
    sudo apt-get install -y libnccl2 libnccl-dev
    ```

3. **Process Launcher** (one of the following):
    - **SLURM**: `srun` command (available on HPC clusters)
    - **Open MPI**: `mpirun` command
      ```bash
      sudo apt-get install -y openmpi-bin libopenmpi-dev
      ```

## Running the sample

1.  Compile this sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/). The binary named `sample_dist_collective` will be created in the `<TensorRT root directory>/bin` directory.

2.  Run the sample with 2 processes. The sample requires the following environment variables:

    - `TRT_MY_RANK`: The rank of this process (0 to WORLD_SIZE-1).
    - `TRT_WORLD_SIZE`: The total number of processes.
    - `TRT_NCCL_ID_FILE`: Path to a shared file for NCCL ID coordination. Rank 0 writes the NCCL unique ID to this file, and other ranks read from it. The file should be empty or non-existent before starting.

    **Using SLURM (srun):**
    ```bash
    srun --ntasks=2 bash -lc 'export TRT_MY_RANK=$SLURM_PROCID; \
        export TRT_WORLD_SIZE=$SLURM_NTASKS; \
        export TRT_NCCL_ID_FILE=/tmp/nccl_id.txt; \
        ./sample_dist_collective --op all_reduce'
    ```

    **Using Open MPI (mpirun):**
    ```bash
    mpirun -np 2 bash -lc 'export TRT_MY_RANK=$OMPI_COMM_WORLD_RANK; \
        export TRT_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE; \
        export TRT_NCCL_ID_FILE=/tmp/nccl_id.txt; \
        ./sample_dist_collective --op all_reduce'
    ```

    **Note:** Make sure to delete or clear the `TRT_NCCL_ID_FILE` before each run to ensure a fresh NCCL ID is generated.

    Available operations:
    - `all_reduce` - Reduces data across all ranks and distributes the result to all ranks
    - `all_gather` - Gathers data from all ranks and distributes the concatenated result to all ranks
    - `broadcast` - Broadcasts data from rank 0 to all other ranks
    - `reduce` - Reduces data across all ranks and sends the result to rank 0
    - `reduce_scatter` - Reduces data across all ranks and scatters portions of the result to each rank

3. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
    ```
    [I] Rank 0 - Generated NCCL ID and wrote to file: /tmp/nccl_id.txt
    [I] Rank 1 - Read NCCL ID from file: /tmp/nccl_id.txt
    [I] Rank 0 - ALL_REDUCE PASSED
    [I] Rank 1 - ALL_REDUCE PASSED
    [I] Rank 0 - ALL_REDUCE test completed successfully!
    [I] Rank 1 - ALL_REDUCE test completed successfully!
    ```

    This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


## Additional resources

The following resources provide a deeper understanding about distributed computing with TensorRT:

**Documentation**
- [Introduction To NVIDIA's TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA's TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)
- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


## Changelog

January 2026
- Initial release of `sampleDistCollective`.


## Known issues

There are no known issues with this sample.
