# TensorRT Python Sample: Stream Writer

This sample demonstrates how to use the TensorRT Python API to serialize an engine directly to a custom stream using the `IStreamWriter` interface, rather than to a file or in-memory buffer. This is useful for advanced scenarios where you want to control how and where the engine bytes are written (e.g., to a network socket, custom buffer, or in-memory stream).

## What does this sample do?

- Builds a simple TensorRT network with two convolutional layers and ReLU activations.
- Implements a custom `StreamWriter` class inheriting from `trt.IStreamWriter` to collect serialized engine bytes.
- Serializes the engine using `builder.build_serialized_network_to_stream()` and writes the bytes to the custom stream.
- Deserializes the engine from the collected bytes to verify correctness.

## File Structure

- `build.py`: Main script containing the sample code.
- `README.md`: This document.

## How to Run

1. **Install Requirements**

   Make sure you have the following Python packages installed:
   - `tensorrt`
   - `numpy`
   - `polygraphy`

   You can install Polygraphy via pip:
   ```
   pip install polygraphy
   ```

   The `tensorrt` Python package is typically provided by NVIDIA as a wheel file.

2. **Run the Sample**

   ```
   python3 build.py
   ```

   You should see output indicating the network is constructed, the engine is built and serialized to the stream, and then deserialized successfully.

## Key Concepts

- **IStreamWriter**: An interface in TensorRT that allows you to define custom logic for writing serialized engine bytes. You must implement the `write(self, data)` method.
- **build_serialized_network_to_stream**: A method that serializes the network and writes the bytes to the provided `IStreamWriter` instance.

## Example Output

```
Constructing network...
[I] TF32 is disabled by default. Turn on TF32 for better performance with minor accuracy differences.
[I] Configuring with profiles:[
        Profile 0:
            {input [min=[1, 3, 224, 224], opt=[1, 3, 224, 224], max=[1, 3, 224, 224]]}
    ]
Building engine and serializing to stream...
The total bytes written to stream is  267836
Deserializing engine from stream...
Engine deserialized successfully
```

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

September 2025
Initial release of this sample.

# Known issues

There are no known issues in this sample.
