## process_engine.py

`process_engine.py` is used to:
1. Build a TensorRT engine from an ONNX file.
2. Profile an engine plan file.
3. Draw an SVG graph from an engine

```
usage: process_engine.py [-h] [--print_only] [--build_engine] [--profile_engine] [--draw_engine] input outdir [trtexec [trtexec ...]]

Utility to build and profile TensorRT engines

positional arguments:
  input                 input file (ONNX or engine)
  outdir                directory to store output artifacts
  trtexec               trtexec commands not including the preceding -- (e.g. int8 shapes=input_ids:32x512,attention_mask:32x512

optional arguments:
  -h, --help            show this help message and exit
  --print_only          print the command-line and exit
  --build_engine, -b    build the engine
  --profile_engine, -p  engine the engine
  --draw_engine, -d     draw the engine
```