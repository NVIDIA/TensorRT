# TensorRT to Triton

This quick start guide showcases how to deploy a simple ResNet model accelerated by using TensorRT on Triton Inference Server. Optimization and deployment go hand in hand in a discussion about Machine Learning infrastructure. For a TensorRT user, network level optimzation to get the maximum performance would already be an area of expertize. 

However, serving this optimized model comes with it's own set of considerations and challenges like: building an infrastructure to support concorrent model executions, supporting clients over HTTP, gRPC and more.

The [Triton Inference Server](https://github.com/triton-inference-server/server) solves the aforementioned and more. Let's discuss step-by-step, the process of optimizing a model with Torch-TensorRT, deploying it on Triton Inference Server, and building a client to query the model. 


## Step 1: Optimize your model with TensorRT

If you are unfamiliar with TensorRT, please refer this [video](https://youtu.be/rK-jxPPY9V4). The first step in this pipeline is to accelerate your model with TensorRT. For the purposes of this demonstration, we are going to assume that you have your trained model in ONNX format. 

(Optional) If you don't have an ONNX model handy and just want to follow along, feel free to use this script:
```
# <xx.xx> is the yy:mm for the publishing tag for NVIDIA's TensorRT 
# container; eg. 22.04

docker run -it --gpus all -v /path/to/this/folder:/resnet50_eg nvcr.io/nvidia/pytorch:<xx.xx>-py3

python export_resnet_to_onnx.py
exit
```

You may need to create an account and get the API key from [here](https://ngc.nvidia.com/setup/). Sign up and login with your key (follow the instructions [here](https://ngc.nvidia.com/setup/api-key) after signing up).

Now that we have an ONNX model, we can use TensorRT to optimize your model. These optimizations are stored in the form of a TensorRT Engine, also known as a TensorRT plan file.

While there are several ways of installing TensorRT, the easiest way is to simply get our pre-built docker container.

```
docker run -it --gpus all -v /path/to/this/folder:/trt_optimize nvcr.io/nvidia/tensorrt:<xx:yy>-py3
```
There are several ways to build a TensorRT Engine; for this demonstration, we will simply use the `trtexec` [CLI Tool](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec).

```
trtexec --onnx=resnet50.onnx \
        --saveEngine=model.plan \
        --explicitBatch \
        --useCudaGraph
```

Before we proceed to the next step, it is important that we know the names of the "input" and "output" layers of your network, as these would be required by Triton. One easy way is to use `polygraphy` which comes packaged with the TensorRT container. If you want to learn more about Polygraphy and its usage, visit [this](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) repository. You can checkout a plethora of [examples](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/inspect) demonstrating the utility of Polygraphy to inspect models.

```
polygraphy inspect model model.plan --mode=basic
```
A section of the output mode looks like this:
```
[I] ==== TensorRT Engine ====
    Name: Unnamed Network 0 | Explicit Batch Engine

    ---- 1 Engine Input(s) ----
    {input [dtype=float32, shape=(1, 3, 224, 224)]}

    ---- 1 Engine Output(s) ----
    {output [dtype=float32, shape=(1, 1000)]}

    ---- Memory ----
    Device Memory: 7225344 bytes

    ---- 1 Profile(s) (2 Binding(s) Each) ----
    - Profile: 0
        Binding Index: 0 (Input)  [Name: input]  | Shapes: min=(1, 3, 224, 224), opt=(1, 3, 224, 224), max=(1, 3, 224, 224)
        Binding Index: 1 (Output) [Name: output] | Shape: (1, 1000)
```

With this, we are ready to proceed to the next step; setting up the Triton Inference Server.

## Step 2: Set Up Triton Inference Server

If you are new to the Triton Inference Server and want to learn more, we highly recommend to check out our [Github Repository](https://github.com/triton-inference-server).

To use Triton, we need to make a model repository. A model repository, as the name suggested, is a repository of the models the Inference server hosts. While Triton can serve models from multiple repositories, in this example, we will discuss the simplest possible form of the model repository. To use Triton, we need to make a model repository. The structure of the repository should look something like this:
```
model_repository
|
+-- resnet50
    |
    +-- config.pbxt
    +-- 1
        |
        +-- model.plan
```

There are two files that Triton requires to serve the model: the model itself and a model configuration file which is typically provided in `config.pbtxt`. We provide a sample of a `config.pbtxt`, which you can use for this specific example. The `config.pbtxt` file is used to describe the exact model configuration with details like the names and shapes of the input and output layer(s), datatypes, scheduling and batching details and more. If you are new to Triton, we highly encourage you to check out this [section of our documentation](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md) for more. 

Once you have the model repository setup, it is time to launch the triton server. You can do that with the docker command below.
```
# Make sure that the TensorRT version in the Triton container
# and TensorRT version in the environment used to optimize the model
# are the same. <xx.yy> as 22.04 will work in this example


docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models
```

## Step 3: Using a Triton Client to Query the Server

Before proceeding, make sure to have a sample image on hand. If you don't have one, download an example image to test inference. In this section, we will be going over a very basic client. For a variety of more fleshed out examples, refer to the [Triton Client Repository](https://github.com/triton-inference-server/client/tree/main/src/python/examples).

```
wget  -O img1.jpg "https://www.hakaimagazine.com/wp-content/uploads/header-gulf-birds.jpg"
```

Install dependencies.
```
pip install torchvision
pip install attrdict
pip install nvidia-pyindex
pip install tritonclient[all]
```

Building a client requires three basic points. 
* Firstly, we setup a connection with the Triton Inference Server. 
* Secondly, we specify the names of the input and output layer(s) of our model.
* Lastly, we send an inference request to the Triton Inference Server.

You can find the corresponding functions of the same in the sample client.
```
python3 triton_client.py
```
The output of the same should look like below:
```
[b'12.472842:90' b'11.523070:92' b'9.660665:14' b'8.407766:136'
 b'8.222099:11']
```
The output format here is `<confidence_score>:<classification_index>`. To learn how to map these to the label names and more, refer to our [documentation](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md).
