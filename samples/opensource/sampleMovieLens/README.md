# Movie Recommendation Using Neural Collaborative Filter (NCF)


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

This sample, sampleMovieLens, is an end-to-end sample that imports a trained TensorFlow model and predicts the highest rated movie for each user. This sample demonstrates a simple movie recommender system using a multi-layer perceptron (MLP) based Neural Collaborative Filter (NCF) recommender.

Specifically, this sample demonstrates how to generate weights for a MovieLens dataset that TensorRT can then accelerate.

## How does this sample work?

The network is trained in TensorFlow on the [MovieLens dataset](https://grouplens.org/datasets/movielens/) containing 6,040 users and 3,706 movies. The NCF recommender system is based off of the [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) paper.

Each query to the network consists of a `userID` and list of `MovieIDs`. The network predicts the highest-rated movie for each user. As trained parameters, the network has embeddings for users and movies, and weights for a sequence of MLPs.

Specifically, this sample:
-   [Imports a network to TensorRT](#importing-a-network-to-tensorrt)
-   [Runs inference](#running-inference)
-   [Verifies the output](#verifying-the-output)

### Importing a network to TensorRT

The network is converted from Tensorflow using the UFF converter (see [Converting A Frozen Graph To UFF](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#samplecode3)), and imported using the UFF parser. Constant layers are used to represent the trained parameters within the network, and the MLPs are implemented using MatrixMultiply layers. A TopK operation is added manually after parsing to find the highest rated movie for the given user.

### Running inference

The sample fills the input buffer with `userIDs` and their corresponding lists of `MovieIDs`, which are loaded from `movielens_ratings.txt`. Then, it launches the inference to predict the rating probabilities for the movies using TensorRT.

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

[TopK layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#topk-layer)
The TopK layer finds the top `K` maximum (or minimum) elements along a dimension, returning a reduced tensor and a tensor of index positions.

## Training an NCF network

This sample comes with a pre-trained model. However, if you want to train your own model, you would need to also convert the model weights to UFF format before you can run the sample.

1.  Clone the NCF repository.
    ```
    git clone https://github.com/hexiangnan/neural_collaborative_filtering.git
    cd neural_collaborative_filtering
    git checkout 0cd2681598507f1cc26d110083327069963f4433
    ```
2.  Apply the `sampleMovieLensTraining.patch` file to save the final result.
    ```
    patch -l -p1 < <TensorRT Install>/samples/sampleMovieLens/sampleMovieLensTraining.patch
    ```
3.  [Install Python 3](https://www.tensorflow.org/install/pip#1.-install-the-python-development-environment-on-your-system).
4. Train the MLP based NCF network.
    ```
    python3 MLP.py --dataset ml-1m --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
    ```

    This step produces the following files in the root directory of the Git repo:
    -   `movielens_ratings.txt`: A text file which contains the lists of `MovieIDs` for each user and the 10 highest-rated `MovieIDs` with their probabilities.
    -   `sampleMovieLens.pb`: The frozen TensorFlow graph which contains the information of the network structure and parameters.

5.  Convert the trained model weights to UFF format which sampleMovieLens understands.
    1.  Convert the `frozen .pb` file to `.uff` format.
        ```
        convert-to-uff sampleMovieLens.pb -p preprocess.py
        ```

        The `preprocess.py` script is a preprocessing step that needs to be applied to the TensorFlow graph before it can be used by TensorRT. The reason for this is that TensorFlow's concatenation operation accounts for the batch dimension while TensorRT's concatenation operation does not.

        The `convert-to-uff` tool is installed together with UFF installation. If you install UFF with deb/rpm, please use the `convert_to_uff.py` script located in `/usr/lib/python3.X/dist-packages/uff*/bin`.

    2.  Copy:
        -   The `sampleMovieLens.uff` file to the `<TensorRT Install>/data/movielens` directory.
        -   The `movielens_ratings.txt` file to the `<TensorRT Install>/data/movielens` directory.


## Running the sample

1. Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleMovieLens` directory. The binary named `sample_movielens` will be created in the `<TensorRT root directory>/bin` directory.
    ```
    cd <TensorRT root directory>/samples/sampleMovieLens
    make
    ```
    Where `<TensorRT root directory>` is where you installed TensorRT.

2. Run the sample to predict the highest-rated movie for each user.
    ```
    cd <TensorRT Install>/bin
    ./sample_movielens # Run with default batch=32 i.e. num of users
    ./sample_movielens -b <N> # Run with batch=N i.e. num of users
    ./sample_movielens --verbose # Prints out inputs, outputs, expected outputs, and expected vs predicted probabilities
    ```

3. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
    ```
    &&&& RUNNING TensorRT.sample_movielens # ./sample_movielens -b 5
    [I] data/movielens/movielens_ratings.txt
    [I] Begin parsing model...
    [I] End parsing model...
    [I] End building engine...
    [I] Done execution. Duration : 514.272 microseconds.
    [I] Num of users : 5
    [I] Num of Movies : 100
    [I] | User : 0 | Expected Item : 128 | Predicted Item : 128 |
    [I] | User : 1 | Expected Item : 133 | Predicted Item : 133 |
    [I] | User : 2 | Expected Item : 515 | Predicted Item : 515 |
    [I] | User : 3 | Expected Item : 23 | Predicted Item : 23 |
    [I] | User : 4 | Expected Item : 134 | Predicted Item : 134 |
    &&&& PASSED TensorRT.sample_movielens # ./sample_movielens -b 5
    ```



    This output shows that the sample ran successfully; `PASSED`.


### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
Usage: ./sample_movielens [-h or --help] [-b NUM_USERS] [--useDLACore=<int>] [--verbose]
--help          Display help information.
--verbose       Enable verbose prints.
-b NUM_USERS    Number of Users i.e. Batch Size (default numUsers==32).
--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.
--fp16          Run in FP16 mode.
--strict        Run with strict type constraints.
```

# Additional resources

The following resources provide a deeper understanding about sampleMovieLens:

**MovieLens**
- [MovieLens dataset](https://grouplens.org/datasets/movielens/)
- [Neural Collaborative Filtering Paper](https://arxiv.org/abs/1708.05031)

**Models**
- [Neural Collaborative Filtering GitHub Repo](https://github.com/hexiangnan/neural_collaborative_filtering)

**Blogs**
- [Accelerating Recommendation System Inference Performance with TensorRT](https://devblogs.nvidia.com/accelerating-recommendation-system-inference-performance-with-tensorrt/)

**Videos**
- [SampleMovieLens YouTube Tutorial](https://www.youtube.com/watch?v=r4KG3dehF48)

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
