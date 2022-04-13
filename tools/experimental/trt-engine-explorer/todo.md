# Todo

## General and usability
* Improve testing.
* Add metadata header to engine file (incl. JSON files + environment desc)
* Add engine build configuration JSON.
* Parse (or add to JSON) timing information (e.g. qps - queries per second)
* Convert to Dash app
* Support Jupyterlab and mito
* Add a command-line interface (to remove scripts)
* Full engine visibility:
  * Integrate with polygraphy (activation stats visualization)
  * Weights distributions (before and after quantization)
* Add educational content
* Add to PyPI
* Add Python engine build and inference from onnx, tf, pytorch
* Capture tactics timing information

## API
* Explain "dataframe as a datastructure"
* Write script to create summary report from engine JSON (PDF/notebook)

## Lint
* Convolution lint rules: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
* Perf hints from: https://docs.google.com/document/d/1-RihnXrUSTSaZUtWgVZXfZHO084PV0XRv3pcm7k8rtI/edit#heading=h.kvi3a99wgs1c
* Convolution lint: flag "non-standard" padding.  E.g. in 7x7 we expect padding of [3,3]
* Scale lint: flag unfused BN layers
* QDQ lint: look for unquantized skip-connections
* Add Shuffle lint
* Warn on small batch size

## Diagrams
* Stacked bars of weights + acts
* show the count per-precision per-layer-type (as a stacked bar)

## Graph rendering
* Add to the repo the code for conversion of engine to ONNX file.
* Engine DOT: add annotations (lint events, latency, highlight reformats, etc)
* Engine DOT: add option to draw node vicinty only

## Engine
* Graph weights distribution (int8 bias detection)
* Report weights sparsity

## Multi-engine comparison
* Be able to visually compare the per-layer performance of two models from different versions of TRT (or from different configs of TRT, like fp32 vs int8).
* Trace engine layers to the original Network or ONNX layers.

## process_engine.py
* Improve the script and document it (e.g. allow timing of several batch configurations with one script invocation).
* Automate comparing perf at several batch sizes

## Debug:
* Make sure that weights, activations match build info

## BKM
* Add a script to lock clocks
