# **Installation**

## **Docker**

Latest TensorFlow 2.x [docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow/tags) from NGC is recommended.

Clone the `tensorflow-quantization` repository, pull the docker image, and launch the container.

```{eval-rst}
    .. code:: console

        $ cd ~/
        $ git clone https://github.com/NVIDIA/TensorRT.git
        $ docker pull nvcr.io/nvidia/tensorflow:22.03-tf2-py3
        $ docker run -it --runtime=nvidia --gpus all --net host -v ~/TensorRT/tools/tensorflow-quantization:/home/tensorflow-quantization nvcr.io/nvidia/tensorflow:22.03-tf2-py3 /bin/bash 
```

After the last command, you will be placed in the `/workspace` directory inside the running docker container, whereas the `tensorflow-quantization` repository is mounted in the `/home` directory.

```{eval-rst}
    .. code:: console

        $ cd /home/tensorflow-quantization
        $ ./install.sh
        $ cd tests
        $ python3 -m pytest quantize_test.py -rP

```

If all tests pass, installation is successful.

## **Local**

```{eval-rst}

.. code:: console

    $ cd ~/
    $ git clone https://github.com/NVIDIA/TensorRT.git
    $ cd TensorRT/tools/tensorflow-quantization
    $ ./install.sh
    $ cd tests
    $ python3 -m pytest quantize_test.py -rP 
```

If all tests pass, installation is successful.
