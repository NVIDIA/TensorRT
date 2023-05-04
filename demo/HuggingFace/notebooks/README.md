# TensorRT Demo with HuggingFace Models

To run the demo Jupyter notebooks in this folder, follow the instructions in the [TRT setup guide](../../../README.md) to build and launch the docker container, e.g. `./docker/build.sh --file docker/ubuntu-20.04.Dockerfile --tag tensorrt-ubuntu20.04-cuda11.7` and `./docker/launch.sh --tag tensorrt-ubuntu20.04-cuda11.7 --gpus all --jupyter <port>` by specifying the port number.

Then, use your browser to start the Jupyter lab interface by opening the token-protected link provided in the terminal, e.g. `http://<host_name>:<port>/lab?token=...`.

Notebook list:

- [gpt2.ipynb](gpt2.ipynb): Step by step walkthrough for building the GPT-2 TensorRT engine.
- [t5.ipynb](t5.ipynb): Step by step walkthrough for building the T5 TensorRT engine.
- [bart.ipynb](bart.ipynb): Step by step walkthrough for building the BART TensorRT engine.
- [playground.ipynb](playground.ipynb): GUI for running customized inputs for a variety of models
