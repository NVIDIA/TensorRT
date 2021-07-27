# A Custom Stem Detection Plugin

Our StemRCNN model is a fork of the MaskRCNN with 2 additional outputs : 
* the stem position (2 dimensions)
* the `is_outside` boolean (1 dimension)

We don't use the `is_outside` prediction anymore so won't need it in the TensorRT version of our model.
But we really need to predict the position of the stem for the entire stack to run normally. The 2 additional outputs
are given by the `Detection Layer`.

As not all the Keras / Tensorflow operations are available in the native TensorRT code, this library comes with samples
that have been developed by the community. The `Detection Layer` of the MaskRCNN is one of them.

Now that we successfully ran our StemRCNN without thes 2 outputs, we would like to run it with the stem position as well.
To do so, we copied [the code of this Detection Layer](https://github.com/NVIDIA/TensorRT/tree/v7.0.0/plugin/detectionLayerPlugin)
and adapt it in order to have this new output. You can see the difference between our version and the original version in TensorRT 
[with this commit](https://github.com/FarmWise/farmwise_cv/commit/597f852688b16696e8e4ef75b7626a67dac19e45)

## Issues when building

Steps to reproduce : 

1. Copy the present folder on the jetson
1. copy the following `.so` files in `lib/TensorRT-7.1.3.0` :
    * libnvinfer_plugin.so
    * libnvinfer_plugin.so.7
    * libnvinfer_plugin.so.7.1.3
    * libnvinfer.so
    * libnvinfer.so.7
    * libnvinfer.so.7.1.3
1. create a `build` folder and move in it
1. Run `cmake ..`
1. Run `make`

You should get the following kind of errors concerning the `cub` library :

```
[ 33%] Building NVCC (Device) object CMakeFiles/tensorrt-custom.dir/src/tensorrt-custom_generated_maskRCNNKernels.cu.o
[...]

/usr/local/cuda-10.2/targets/aarch64-linux/include/thrust/system/cuda/detail/cub/device/dispatch/../../agent/agent_reduce_by_key.cuh(116): error: type name is not allowed

/usr/local/cuda-10.2/targets/aarch64-linux/include/thrust/system/cuda/detail/cub/device/dispatch/../../agent/agent_reduce_by_key.cuh(116): error: the global scope has no "VALUE"

Error limit reached.
100 errors detected in the compilation of "/tmp/tmpxft_00005e9d_00000000-9_maskRCNNKernels.compute_75.cpp1.ii".
Compilation terminated.
CMake Error at tensorrt-custom_generated_maskRCNNKernels.cu.o.cmake:280 (message):
  Error generating file
  /home/engineering/workspace/plugins/stemDetectionLayerPlugin/build/CMakeFiles/tensorrt-custom.dir/src/./tensorrt-custom_generated_maskRCNNKernels.cu.o


CMakeFiles/tensorrt-custom.dir/build.make:75: recipe for target 'CMakeFiles/tensorrt-custom.dir/src/tensorrt-custom_generated_maskRCNNKernels.cu.o' failed
make[2]: *** [CMakeFiles/tensorrt-custom.dir/src/tensorrt-custom_generated_maskRCNNKernels.cu.o] Error 1
CMakeFiles/Makefile2:82: recipe for target 'CMakeFiles/tensorrt-custom.dir/all' failed
make[1]: *** [CMakeFiles/tensorrt-custom.dir/all] Error 2
Makefile:90: recipe for target 'all' failed
make: *** [all] Error 2
```

Without the following line in `CMakeLists.txt`, the error says that it cannot find the file `cub/cub.cuh` :
```
include_directories("/usr/local/cuda-10.2/targets/aarch64-linux/include/thrust/system/cuda/detail")
```

Here is the error : 

```
[ 33%] Building NVCC (Device) object CMakeFiles/tensorrt-custom.dir/src/tensorrt-custom_generated_maskRCNNKernels.cu.o
/home/engineering/workspace/plugins/stemDetectionLayerPlugin/src/maskRCNNKernels.cu:26:10: fatal error: cub/cub.cuh: No such file or directory
 #include <cub/cub.cuh>
          ^~~~~~~~~~~~~
compilation terminated.
CMake Error at tensorrt-custom_generated_maskRCNNKernels.cu.o.cmake:220 (message):
  Error generating
  /home/engineering/workspace/plugins/stemDetectionLayerPlugin/build/CMakeFiles/tensorrt-custom.dir/src/./tensorrt-custom_generated_maskRCNNKernels.cu.o
```

## Interesting readings

* [NVIDIA documentation about plugins](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#add_custom_layer)
* [A thread on a similar issue](https://forums.developer.nvidia.com/t/unable-to-include-cub/61743)
