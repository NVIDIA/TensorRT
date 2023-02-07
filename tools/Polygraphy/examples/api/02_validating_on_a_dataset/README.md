# Validating Accuracy On A Real Dataset


## Introduction

The `Comparator` provided by Polygraphy can be useful for comparing a small number of
results across multiple runners, but is not well suited for validating a single runner
with a real dataset that includes labels or golden values - especially if the dataset is large.

In such cases, it is recommended to use a runner directly instead.

*NOTE: It is possible to provide custom input data to `Comparator.run()` using the `data_loader`*
    *parameter. This may be a viable option when using a smaller dataset.*

In this example, we use a `TrtRunner` directly to validate an identity model on
a trivial dataset. Unlike using the `Comparator`, using a runner gives you complete
freedom as to how you load your input data, as well as how you validate the results.

Since all runners provide the same interface, you can freely drop-in other runners
without touching the rest of your validation code. For example, in this case, validating
the model using ONNX-Runtime would require changing just 2 lines; this is left as an
exercise for the reader.


## Running The Example

1. Install prerequisites
    * Ensure that TensorRT is installed
    * Install other dependencies with `python3 -m pip install -r requirements.txt`

2. Run the example
    ```bash
    python3 example.py
    ```
