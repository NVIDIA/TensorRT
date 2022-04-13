# trt-engine-explorer

![](images/trex.png)

This repository contains Python code (`trex` package) to explore various aspects of a TensorRT engine plan and its associated inference profiling data.

An engine plan file is a serialized TensorRT engine format. It contains information about the final inference graph and can be deserialized for inference runtime execution.  An engine plan is specific to the hardware and software versions of the system used to build the engine.

`trex` is useful for initial model performance debugging, visualization of plan graphs, and for understanding the characteristics of an engine plan. <b>For in-depth performance analysis, [Nvidia &reg; Nsight Systems &trade;](https://developer.nvidia.com/nsight-systems) is the recommended performance analysis tool.
# Features
The `trex` package contains an API and Jupyter notebooks for viewing and inspecting TensorRT engine-plan files and profiling data.

* An engine plan graph (JSON) is loaded to a Pandas dataframe which allows slicing, querying, filtering, viewing and diagraming.
* An engine plan graph can be visualized as SVG/PNG files.
* Layer linters are an API for flagging potential performance hazards (preview feature).
* Three Jupyter notebooks provide:
  * `trex` API tutorial.
  * Detailed engine plan performance, characteristics and structure analysis.
  * Comparison of two or more engine plans.
* Because `trex` operates on JSON input files, it does not require a GPU.

## Caveats
When `trtexec` times individual layers, the total latency (computed by summing the average latency of each layer) is higher than the latency reported for the entire engine.
## Supported TenorRT Versions
Starting with TensorRT 8.2, engine-plan graph and profiling data can be exported to JSON files. `trex` supports TensortRT 8.2 and 8.4.

`trex` has only been tested on Ubuntu 18.04 LTS, with Python 3.6. `trex` does not require a GPU, but generating the input JSON file(s) does require a GPU.

<details><summary><h1>Installation</h1></summary>


The instructions below detail how to use a Python3 virtualenv for installing and using trex (Python 3.6+ is required).

### 1. Clone the trex code repository from TensorRT OSS repository
```
$ git clone https://github.com/NVIDIA/TensorRT.git
```

### 2. Create and activate a Python virtual environment
The commands listed below create and activate a Python virtual enviornment named ```env_trex``` which is stored in a directory by the same name, and configures the current shell to use it as the default python environment.

```
$ cd TensorRT/tools/experimental/trt-engine-explorer
$ python3 -m virtualenv env_trex
$ source env_trex/bin/activate
```

### 3. Install trex in development mode and the Jupyter extensions required for the notebooks
```
$ python3 -m pip install -e .
$ jupyter nbextension enable widgetsnbextension --user --py
```

</details>

<details><summary><h1>Generating engine plan and timing JSON files</h1></summary>

You can use the Python script `utils/process_engine.py` to perform several functions:
1. Create an engine from an ONNX file.
2. Load an engine and create an engine-plan JSON file.
3. Profile an engine inference execution and store the results in an engine timing JSON file.
4. Create an engine graph diagram.

![](images/trex-overview.png)

For more information see [TensorRT Engine Inspector](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#engine-inspector) and the [Tutorial](notebooks/tutorial.ipynb) notebook.

</details>

<details><summary><h1>Launching Jupyter</h1></summary>

Launch the Jupyter notebook server as detailed below and open your browser at `http://localhost:8888` or `http://<your-ip-address>:8888`
```
$ jupyter-notebook --ip=0.0.0.0 --no-browser
```

</details>

<details><summary><h1>Changelog</h1></summary>

April 2022: Initial release of this sample

</details>

<details><summary><h1>License</h1></summary>

The TensorRT Engine Explorer license can be found in the [LICENSE](LICENSE.txt) file.

</details>
