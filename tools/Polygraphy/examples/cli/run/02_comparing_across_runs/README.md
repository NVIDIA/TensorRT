# Comparing Across Runs

In some cases, it may be be useful to compare results across different runs of `polygraphy run` -
for example, comparing between different machines, or comparing between multiple different versions
of the same library.

For example, if you need to compare results between two different systems (let's call them
System A and System B), you can first save the results from System A:


```bash
polygraphy run identity.onnx --onnxrt \
    --save-outputs system_a_results.json
```

Next, you can run the model on System B, and load the results saved from
System A to compare against:

```bash
polygraphy run identity.onnx --onnxrt \
    --load-outputs system_a_results.json
```
