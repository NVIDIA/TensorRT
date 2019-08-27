# BERT Python Demo

## Setting Up Your Environment

1. Clone the TensorRT repository and navigate to BERT demo directory
    ```
    git clone --recursive https://github.com/NVIDIA/TensorRT && cd TensorRT/demo/BERT
    ```

2. Create and launch the docker image
    ```
    sh python/create_docker_container.sh
    ```

3. Build the plugins and download the fine-tuned models
    ```
    cd TensorRT/demo/BERT && sh python/build_examples.sh
    ```

## Building an Engine
To build an engine, run the `bert_builder.py` script. For example,
```
python python/bert_builder.py -m /workspace/models/fine-tuned/bert_tf_v2_base_fp16_384_v2/model.ckpt-8144 -o bert_base_384.engine -b 1 -s 384 -c /workspace/models/fine-tuned/bert_tf_v2_base_fp16_384_v2
```
This will build and engine with a maximum batch size of 1 (`-b 1`), and sequence length of 384 (`-s 384`) using the `bert_config.json` file located in `workspace/models/fine-tuned/bert_tf_v2_base_fp16_384_v2`

## Running Inference
Finally, you can run inference with the engine generated from the previous step using the `bert_inference.py` script. This script also accepts a passage and a question. For example,
```
python python/bert_inference.py -e bert_base_384.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v /workspace/models/fine-tuned/bert_tf_v2_base_fp16_384_v2/vocab.txt -b 1
```
