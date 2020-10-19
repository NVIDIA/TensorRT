# Interoperating With TensorRT


## Introduction

A key feature of Polygraphy is complete interoperability with TensorRT, as well as
with other backends. Since Polygraphy does not hide the underlying backend APIs,
it is possible to freely switch between using the Polygraphy API and a backend API,
such as TensorRT.

In this example, we'll look at how you can retain access to the advanced functionality
provided by a backend without giving up the conveniences provided by Polygraphy - the
best of both worlds.

Polygraphy provides an `extend` decorator which can be used to easily extend existing
Polygraphy loaders. This can be useful in many scenarios, but for this example,
we will focus on cases where you may want to:
- Modify the TensorRT network prior to building the engine
- Use a TensorRT builder flag not currently supported by Polygraphy


## Running the Example

1. Install prerequisites
    a. Ensure that TensorRT is installed
    b. Install other dependencies with `python3 -m pip install -r requirements.txt`

2. Run the example
    ```bash
    python3 example.py
    ```
