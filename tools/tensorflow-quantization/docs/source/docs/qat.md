(qat)=

# **Quantization Aware Training (QAT)**

The process of converting continuous to discrete values (Quantization) and vice-versa (Dequantization), requires `scale` and `zeroPt` (zero-point) parameters to be set.
There are two quantization methods based on how these two parameters are calculated:

```{eval-rst}

#. `Post Training Quantization (PTQ) <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_c>`_
    Post Training Quantization computes `scale` after network has been trained. A representative dataset is used to capture the distribution of activations for each activation tensor, then this distribution data is used to compute the `scale` value for each tensor.
    Each weight's distribution is used to compute weight `scale`.

    TensorRT provides a workflow for PTQ, called `calibration`.

    .. mermaid::

        flowchart LR
            id1(Calibration data) --> id2(Pre-trained model) --> id3(Capture layer distribution) --> id4(Compute 'scale') --> id5(Quantize model)

#. `Quantization Aware Training (QAT) <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work-with-qat-networks>`_
    Quantization Aware Training aims at computing scale factors during training. Once the network is fully trained, Quantize (Q) and Dequantize (DQ) nodes are inserted into the graph following a specific set of rules. The network is then further trained for few epochs in a process called `Fine-Tuning`. Q/DQ nodes simulate quantization loss and add it to the training loss during fine-tuning, making the network more resilient to quantization. In other words, QAT is able to better preserve accuracy when compared to PTQ.

    .. mermaid::

        flowchart LR
            id1(Pre-trained model) --> id2(Add Q/DQ nodes) --> id3(Finetune model) --> id4(Store 'scale') --> id5(Quantize model)

```

```{attention}
This toolkit supports only QAT as a quantization method. Note that we follow the quantization algorithm implemented by TensorRT(TM) when inserting Q/DQ nodes in a model. This leads to a quantized network with optimal layer fusion during the TensorRT(TM) engine building step.
```

````{note}
Since TensorRT(TM) only supports symmetric quantization, we assume `zeroPt = 0`.
````
