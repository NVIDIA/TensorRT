#!/bin/bash

docker run -it --rm \
    --name sample-bert \
    --runtime=nvidia \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -u $(id -u):$(id -g) \
    -v ${HOME}:/host/ \
    -v $1:/data/ \
    -v $(pwd)/../../:/workspace/TensorRT \
    sample-bert bash


