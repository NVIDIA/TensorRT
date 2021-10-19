# “Hello World” For TensorRT Using TensorFlow And Python


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [Freezing a TensorFlow graph](#freezing-a-tensorflow-graph)
	* [Freezing a Keras model](#freezing-a-keras-model)
- [Generate the UFF model](#generate-the-uff-model)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, `end_to_end_tensorflow_mnist`, trains a small, fully-connected model on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and runs inference using TensorRT.

## How does this sample work?

This sample is an end-to-end Python sample that trains a [small 3-layer model in TensorFlow and Keras](https://www.tensorflow.org/tutorials), freezes the model and writes it to a protobuf file, converts it to UFF, and finally runs inference using TensorRT.

### Freezing a TensorFlow graph

In order to use the command-line [UFF utility](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/uff/uff.html), TensorFlow graphs must be frozen and saved as `.pb` files.

In this sample, the converter displays information about the input and output nodes, which you can use to the register inputs and outputs with the parser. In this case, we already know the details of the input and output nodes and have included them in the sample.

### Freezing a Keras model

You can use the following sample code to freeze a Keras model.
```
def save(model, filename):
	# First freeze the graph and remove training nodes.
	output_names = model.output.op.name
	sess = tf.keras.backend.get_session()
	frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_names])
	frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
	# Save the model
	with open(filename, "wb") as ofile:
		ofile.write(frozen_graph.SerializeToString())
```

## Generate the UFF model

1. If running this sample in a test container, launch [NVIDIA tf1 (Tensorflow 1.x)](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running) container in a separate terminal for generating the UFF model.
    ```bash
    docker run --rm -it --gpus all -v `pwd`:/workspace nvcr.io/nvidia/tensorflow:21.03-tf1-py3 /bin/bash
    ```

    Alternatively, install Tensorflow 1.15
    `pip3 install tensorflow>=1.15.5,<2.0`

  NOTE
  - On PowerPC systems, you will need to manually install TensorFlow using IBM's [PowerAI](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.0/navigation/pai_install.htm).
  - On Jetson boards, you will need to manually install TensorFlow by following the documentation for [Xavier](https://docs.nvidia.com/deeplearning/dgx/install-tf-xavier/index.html) or [TX2](https://docs.nvidia.com/deeplearning/dgx/install-tf-jetsontx2/index.html).

2. Run the sample to train the model and write out the frozen graph:
    ```bash
    mkdir -p models
    python3 model.py
    ```

3. Install the UFF toolkit and graph surgeon depending on your [TensorRT installation method](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing), or from PyPI:
    ```bash
    pip3 install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com uff
    pip3 install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com graphsurgeon
    ```

4. The MNIST dataset can be found under the data directory (usually `/usr/src/tensorrt/data/mnist`) if using the TensorRT containers. It is also bundled along with the [TensorRT tarball](https://developer.nvidia.com/nvidia-tensorrt-download).

5. Convert the `.pb` file to `.uff` using the convert-to-uff utility:
    ```bash
    convert-to-uff models/lenet5.pb
    ```

    Depending on how you installed TensorRT, this utility may also be located in `/usr/lib/python<PYTHON3 VERSION>/site-packages/uff/bin/convert_to_uff.py`.

## Prerequisites

1. Switch back to test container (if applicable) and install the dependencies for Python.

```bash
pip3 install -r requirements.txt
```

On Jetson Nano, you will need nvcc in the `PATH` for installing pycuda:
```bash
export PATH=${PATH}:/usr/local/cuda/bin/
```

## Running the sample

1.  Create a TensorRT inference engine from the UFF file and run inference:
    ```bash
    python3 sample.py [-d DATA_DIR]
    ```

  * NOTE: If the MNIST image data is not installed in the default location, `/usr/src/tensorrt/data/` as shown, the data directory must be specified.
	For example: `python3 sample.py -d /path/to/my/data/`.

2.  Verify that the sample ran successfully. If the sample runs successfully you should see a match between the test case and the prediction.
	```
	Test Case: 2
	Prediction: 2
	```

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
usage: sample.py [-h] [-d DATADIR]

Runs an MNIST network using a UFF model file

optional arguments:
 -h, --help            show this help message and exit
 -d DATADIR, --datadir DATADIR
                       Location of the TensorRT sample data directory.
                       (default: /usr/src/tensorrt/data)
```

# Additional resources

The following resources provide a deeper understanding about training and running inference in TensorRT using Python:

**Model**
- [TensorFlow/Keras MNIST](https://www.tensorflow.org/tutorials)

**Dataset**
- [MNIST database](http://yann.lecun.com/exdb/mnist/)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

February 2019
This `README.md` file was created.

# Known issues

There are no known issues in this sample.
