# NVIDIA TensorRT Sample "sampleGoogleNet"

The sampleGoogleNet sample demonstrates how to:
- Build a TensorRT engine from the saved Caffe model
- Set input values to engine, run engine and obtain output

## Usage

This sample can be run as:

    ./sample_googlenet [-h] [--datadir=/path/to/data/dir/] [--useDLACore=N]

SampleGoogleNet reads two caffe files to build the network:

* `googlenet.prototxt` - The prototxt file that contains the network design
* `googlenet.caffemodel` - The model file which contains the trained weights
  for the network

By default, the sample expects these files to be in `data/samples/googlenet/` or
`data/googlenet/`. The default directory can be changed by supplying the path as
`--datadir=/new/path/` as a command line argument.
