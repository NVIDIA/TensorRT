#!/bin/bash

python -m pip uninstall --yes tensorflow_quantization
rm -rf  tensorflow_quantization.egg-info
rm -rf build
python -m pip install .