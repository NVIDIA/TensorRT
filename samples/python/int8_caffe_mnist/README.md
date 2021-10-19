# INT8 Calibration In Python


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, `int8_caffe_mnist`, demonstrates how to create an INT8 calibrator, build and calibrate an engine for INT8 mode, and finally run inference in INT8 mode.

## How does this sample work?

During calibration, the calibrator retrieves a total of 1003 batches, with 100 images each. We have simplified the process of reading and writing a calibration cache in Python, so that it is now easily possible to cache calibration data to speed up engine builds (see `calibrator.py` for implementation details).

During inference, the sample loads a random batch from the calibrator, then performs inference on the whole batch of 100 images.

## Prerequisites

1. Install the dependencies for Python.
    ```bash
    pip3 install -r requirements.txt
    ```

On Jetson Nano, you will need nvcc in the `PATH` for installing pycuda:
```bash
export PATH=${PATH}:/usr/local/cuda/bin/
```

2. The MNIST dataset can be found under the data directory (usually `/usr/src/tensorrt/data/mnist`) if using the TensorRT containers. It is also bundled along with the [TensorRT tarball](https://developer.nvidia.com/nvidia-tensorrt-download).
   - This sample requires the [training set](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz), [test set](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz) and [test labels](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz).
   - Unzip these files once they are downloaded - Note that on Windows depending on how you decompress these files the resulting file extention may be changed to <FILENAME>.idx3-ubyte. If this happens, rename the files to <FILENAME>-idx3-ubyte


## Running the sample

1.  Run the sample to create a TensorRT inference engine, perform IN8 calibration and run inference:
	`python3 sample.py [-d DATA_DIR]`

	to run the sample with Python 3.

	**Note:** If the TensorRT sample data is not installed in the default location, for example `/usr/src/tensorrt/data/`, the `data` directory must be specified. For example:
	`python3 sample.py -d /path/to/my/data/`.


2.  Verify that the sample ran successfully. If the sample runs successfully, the accuracy should be close to 99%.

	```
	Calibrating batch 0, containing 64 images
	...
	Calibrating batch 150, containing 64 images
	Validating batch 10
	...
	Validating batch 310
	Total Accuracy: 99.09%

	```

### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.

# Additional resources

The following resources provide a deeper understanding about the model used in this sample:

**Network:**
- [MNIST network](http://yann.lecun.com/exdb/lenet/)

**Dataset:**
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [Enabling INT8 Inference Using Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#enable_int8_python)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

March 2019
This `README.md` file was recreated, updated and reviewed.


# Known issues

There are no known issues in this sample.
