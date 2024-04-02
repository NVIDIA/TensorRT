# trex Command-line Tool

The `trex` command-line tool (not to be confused with the `trex` package) provides a convinient interface to some of the utilities in the `utils` directory. It is installed with the `trex` package.

## trex draw
Draw a graph diagram of a TensorRT engine graph JSON file.<br>

Example:
```
$ trex draw ./examples/pytorch/resnet/A100/fp32/resnet.onnx.engine.graph.json --display_regions --no_layer_names
```

## trex process
Build, profile and draw a TensorRT engine.
```
$ trex process ./examples/pytorch/resnet/generated/resnet.onnx ./examples/pytorch/resnet/A100/fp32/
```