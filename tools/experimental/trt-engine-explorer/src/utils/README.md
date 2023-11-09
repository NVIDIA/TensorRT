# Utility scripts
  * [process_engine.py](#process-enginepy)
  * [draw_engine.py](#draw-enginepy)
  * [parse_trtexec_log.py](#parse-trtexec-logpy)

<br>

## process_engine.py

`process_engine.py` is used to:
1. Build a TensorRT engine from an ONNX file.
2. Profile an engine plan file.
3. Generate JSON files for exploration with trex.
4. Draw an SVG graph from an engine.

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

The script can run the entire ONNX to JSON files pipeline, or it can execute a single sub-command. For example, the following command line builds and profiles an engine from the ONNX model stored in a file named `my_onnx.onnx`:
```
$ process_engine.py my_onnx.onnx outputs_dir int8
```

This will generate the following files in directory `outputs_dir`:
* `my_onnx.onnx.engine` - the built engine file.
* `my_onnx.onnx.engine.build.log` - trtexec engine building log.
* `my_onnx.onnx.engine.build.metadata.json` - JSON of metadata parsed from the build log.
* `my_onnx.onnx.engine.graph.json` - JSON of engine graph.
* `my_onnx.onnx.engine.graph.json.svg` - SVG diagram of engine graph.
* `my_onnx.onnx.engine.profile.json` - JSON of engine layers profiling.
* `my_onnx.onnx.engine.profile.log` - trtexec engine profiling log.
* `my_onnx.onnx.engine.profile.metadata.json` - JSON of metadata parsed from the profiling log.
* `my_onnx.onnx.engine.timing.json` - JSON of engine profiling iteration timing.


Requirements:
* Path to trtexec binary is in $PATH.
* trex is installed (for graph drawing).
* Graphviz is installed (for graph drawing).
```
$ sudo apt-get --yes install graphviz
```

## draw_engine.py

This script generates an SVG diagram of the input engine graph SVG file.

## parse_trtexec_log.py

This script parses `trtexec` log files and creates metadata JSON files from the information extracted from the logs.
