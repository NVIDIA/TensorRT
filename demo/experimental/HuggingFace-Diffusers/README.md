# Introduction

This demo notebook showcases the acceleration of Stable Diffusion pipeline using TensorRT through HuggingFace pipelines.

# Setup

### Clone the TensorRT OSS repository

```bash
git clone git@github.com:NVIDIA/TensorRT.git -b release/9.0 --single-branch
cd TensorRT/demo/experimental/HuggingFace-Diffusers
```

### Launch TensorRT NGC container

Install nvidia-docker using [these intructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). Launch the docker container with the following command:

```bash
docker run --rm -it --gpus all -p 8888:8888 -v $PWD:/workspace nvcr.io/nvidia/tensorrt:23.04-py3 /bin/bash
```

### Run Jupyter Notebook

Install `jupyter` with:

```bash
pip install jupyter
```

Launch the notebook within the container with:

```bash
jupyter notebook --ip 0.0.0.0 TensorRT-diffusers-txt2img.ipynb --allow-root --no-browser
```

Follow the console output for the link to run the notebook on your host machine.
