# Neural Network Driven Framework

NNDF are a collection of files and formats that provide an underlying policy and flow for TensorRT network onboarders to follow.
NNDF is inspired by HuggingFace and PyTorch common design architectures where the Neural Network is divided into two abstractions:

* High level abstractions via configuration files
* Low level abstractions via I/O classes

## Benefits

Because NNDF is inspired by existing successful network frameworks, interoping and interacting with HuggingFace, Torch, and other
networks become very trivial and code can often be reused. See for example the `GenerationMixin` which is used in HuggingFace to
implement `greedy_decoder` and `beam_search`. Using NNDF, we can use `beam_search` and other search functions directly.

In other words:

* Re-use high level measurement tools supplied by well known frameworks
* Ensure fair platform for timing TRT performance alongside other frameworks by using the same post-processing code.
