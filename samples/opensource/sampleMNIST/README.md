# NVIDIA TensorRT Sample "sampleMNIST"

The sampleMNIST sample demonstrates how to:
- Perform the basic setup and initialization of TensorRT
- Import a trained Caffe MNIST digit classifier
- Perform simple input preprocessing before running the engine, like mean
  normalization in this case
- Verify the output from the inference

## Usage

This sample can be run as:

    ./sample_mnist [-h] [--datadir=/path/to/data/dir/] [--useDLA=N]

SampleMNIST reads two Caffe files to build the network:

* `mnist.prototxt` - The prototxt file that contains the network design
* `mnist.caffemodel` - The model file which contains the trained weights
  for the network
* `mnist_mean.binaryproto` - The binaryproto file which contains the means

By default, the sample expects these files to be in `data/samples/mnist/` or
`data/mnist/`. The list of default directories can be changed by adding one or
more paths with `--datadir=/new/path/` as a command line argument.
