#!/bin/bash

# Minimal and backward-compatible: default to modern requirements, allow override via env.
REQ_FILE="${REQUIREMENTS_FILE:-requirements.txt}"
echo "[I] Using requirements file: ${REQ_FILE} (override with REQUIREMENTS_FILE=<file>)"

# Upgrade pip and install TensorRT
python3 -m pip install --upgrade pip
pip3 install --pre tensorrt-cu12

# Install the required packages
PIP_CONSTRAINT= pip3 install -r "${REQ_FILE}"

# Install libgl1
# Check if apt-get is available
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get install -y libgl1
else
    echo "Warning: apt-get not found. Please install libgl1 manually for your system."
fi

