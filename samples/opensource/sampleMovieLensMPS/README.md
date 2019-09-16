# Movie Recommendation Using MPS (Multi-Process Service)


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [Importing a network to TensorRT](#importing-a-network-to-tensorrt)
	* [Running inference](#running-inference)
	* [Verifying the output](#verifying-the-output)
	* [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Training an NCF network](#training-an-ncf-network)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleMovieLensMPS, is an end-to-end sample that imports a trained TensorFlow model and predicts the highest rated movie for each user using MPS (Multi-Process Service).

MPS allows multiple CUDA processes to share single GPU context. With MPS, multiple overlapping kernel execution and `memcpy` operations from different processes can be scheduled concurrently to achieve maximum utilization. This can be especially effective in increasing parallelism for small networks with low resource utilization such as those primarily consisting of a series of small MLPs.

This sample is identical to [sampleMovieLens](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#sample_movie) in terms of functionality, but is modified to support concurrent execution in multiple processes. Specifically, this sample demonstrates how to generate weights for a MovieLens dataset that TensorRT can then accelerate.

**Note:** Currently, sampleMovieLensMPS supports only Linux x86-64 (includes Ubuntu and RedHat) desktop users.

## How does this sample work?

The network is trained in TensorFlow on the [MovieLens dataset](https://grouplens.org/datasets/movielens/) containing 6,040 users and 3,706 movies. The NCF recommender system is based off of the [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) paper.

Each query to the network consists of a `userID` and list of `MovieIDs`. The network predicts the highest-rated movie for each user. As trained parameters, the network has embeddings for users and movies, and weights for a sequence of MLPs.

Specifically, this sample:
-   [Imports a network into TensorRT](#importing-a-network-to-tensorrt)    
-   [Runs the inference](#running-inference)    
-   [Verifies its output](#verifying-the-output)

### Importing a network to TensorRT

The network is converted from Tensorflow using the UFF converter (see [Converting A Frozen Graph To UFF](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#samplecode3)), and imported using the UFF parser. Constant layers are used to represent the trained parameters within the network, and the MLPs are implemented using MatrixMultiply layers. A TopK operation is added manually after parsing to find the highest rated movie for the given user.

### Running inference

The sample fills the input buffer with `userIDs` and their corresponding lists of `MovieIDs`, which are loaded from `movielens_ratings.txt`. Then, it launches the inference to predict the rating probabilities for the movies using TensorRT. The inference will be launched on multiple processes. When MPS is enabled, the processes will share one single CUDA context to reduce context overhead. See [Multi-Process Service Introduction](https://docs.nvidia.com/deploy/mps/index.html) for more details about MPS.

### Verifying the output

Finally, the sample compares the outputs predicted by TensorRT with the expected outputs which are given by `movielens_ratings.txt`. For each user, the `MovieID` with the highest probability should match the expected highest-rated `MovieID`. In the verbose mode, the sample also prints out the probability, which should be close to the expected probability.

### TensorRT API layers and ops

In this sample, the following layers are used.  For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions.

[MatrixMultiply layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#matrixmultiply-layer)
The MatrixMultiply layer implements matrix multiplication for a collection of matrices.

[Scale layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#scale-layer)
The Scale layer implements a per-tensor, per-channel, or per-element affine transformation and/or exponentiation by constant values.

[Shuffle layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#shuffle-layer)
The Shuffle layer implements a reshape and transpose operator for tensors.

### TensorRT API layers and ops

In this sample, the following layers are used.  For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions.

[MatrixMultiply layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#matrixmultiply-layer)
The MatrixMultiply layer implements matrix multiplication for a collection of matrices.

[Scale layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#scale-layer)
The Scale layer implements a per-tensor, per-channel, or per-element affine transformation and/or exponentiation by constant values.

[Shuffle layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#shuffle-layer)
The Shuffle layer implements a reshape and transpose operator for tensors.

[TopK layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#topk-layer)
The TopK layer finds the top `K` maximum (or minimum) elements along a dimension, returning a reduced tensor and a tensor of index positions.

## Training an NCF network

This sample comes with a pre-trained model. However, if you want to train your own model, you would need to also convert the model weights to UFF format before you can run the sample. For step-by-step instructions, refer to the `README.md` file in the `sampleMovieLens` directory.

## Running the sample

1. Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleMovieLensMPS` directory. The binary named `sample_movielens_mps` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples/sampleMovieLensMPS
	make
	```
	Where `<TensorRT root directory>` is where you installed TensorRT.


2. Set-up an MPS server.
	```
	export CUDA_VISIBLE_DEVICES=<GPU_ID>
	nvidia-smi -i <GPU_ID> -c EXCLUSIVE_PROCESSexport CUDA_VISIBLE_DEVICES=0
	export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # Select a location that's accessible to the given $UID
	export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log # Select a location that's accessible to the given $UID
	nvidia-cuda-mps-control -d # Start the daemon.
	```
	The log files of MPS are located at:
	```
	$CUDA_MPS_LOG_DIRECTORY/control.log
	$CUDA_MPS_LOG_DIRECTORY/server.log
	```
3. Set-up an MPS client. Set the following variables in the client process environment. The `CUDA_VISIBLE_DEVICES` variable should not be set in the client's environment.
	```
	export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # Set to the same location as the MPS control daemon
	export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log # Set to the same location as the MPS control daemon
	```
4. Run the sample from an MPS client to predict the highest-rated movie for each user on multiple processes.
	```
	cd <TensorRT Install>/bin
	./sample_movielens_mps (default batch=32 i.e. num of users, Number of processes=1)
	./sample_movielens_mps -b <bSize> -p <nbProc> (bSize=Batch size i.e. num of users, nbProc=Number of processes)
	./sample_movielens_mps --verbose (prints inputs, groundtruth values, expected vs predicted probabilities)
	```
5. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_movielens_mps # build/cuda-		10.0/7.3/x86_64/sample_movielens_mps -b 2 -p 2
	[I] data/samples/movielens/movielens_ratings.txt
	[I] Begin parsing model...
	[I] End parsing model...
	[I] End building engine...
	[I] Done execution in process: 24136 . Duration : 214.272 microseconds.
	[I] Num of users : 2
	[I] Num of Movies : 100
	[I] | PID : 24136 | User : 0 | Expected Item : 128 | Predicted Item : 128 |
	[I] | PID : 24136 | User : 1 | Expected Item : 133 | Predicted Item : 133 |
	[I] Done execution in process: 24135 . Duration : 214.176 microseconds.
	[I] Num of users : 2
	[I] Num of Movies : 100
	[I] | PID : 24135 | User : 0 | Expected Item : 128 | Predicted Item : 128 |
	[I] | PID : 24135 | User : 1 | Expected Item : 133 | Predicted Item : 133 |
	[I] Number of processes executed: 2. Number of processes failed: 0.
	[I] Total MPS Run Duration: 1737.51 milliseconds.
	&&&& PASSED TensorRT.sample_movielens_mps # build/cuda-	10.0/7.3/x86_64/sample_movielens_mps -b 2 -p 2
	```
	This output shows that the sample ran successfully; `PASSED`. The output also shows that the 	predicted items for each user matches the expected items and the duration of the execution. Finally, the sample prints out the PIDs of the processes, showing that the inference is launched on multiple processes.

6. To restore the system to its original state, shutdown MPS, if needed.
	`echo quit | nvidia-cuda-mps-control`

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
Usage:
         ./sample_movielens_mps [-h] [-b NUM_USERS] [-p NUM_PROCESSES] [--useDLACore=<int>] [--verbose]
        -h             Display help information. All single dash options enable perf mode.
        -b             Number of Users i.e. Batch Size (default numUsers=32).
        -p             Number of child processes to launch (default nbProcesses=1. Using MPS with this option is strongly recommended).
        --useDLACore=N Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.
        --verbose      Enable verbose prints.
        --int8         Run in Int8 mode.
        --fp16         Run in FP16 mode.
```

# Additional resources

The following resources provide a deeper understanding about sampleMovieLensMPS:

**MovieLensMPS**
- [MovieLens dataset](https://grouplens.org/datasets/movielens/)
- [Neural Collaborative Filtering Paper](https://arxiv.org/abs/1708.05031)
- [Multi-Process Service Introduction](https://docs.nvidia.com/deploy/mps/index.html)

**Models**
- [Neural Collaborative Filtering GitHub Repo](https://github.com/hexiangnan/neural_collaborative_filtering)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Jupyter Notebook Tutorial for SampleMovieLens](https://developer.download.nvidia.com/compute/machine-learning/tensorrt/models/sampleMLP-notebook.html?ncid=--47568)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

February 2019
This `README.md` file was recreated, updated and reviewed.


# Known issues

- Since the UFF converter is not currently supported on Windows, the model cannot be converted to UFF on Windows systems. It is still possible to use the UFF file shipped with the sample.
