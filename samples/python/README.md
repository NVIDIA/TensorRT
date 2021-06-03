# General Setup for Python Samples

## Prerequisites

Dependencies can be istalled using:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

Data can be downloaded using the following utility if `download.yml` is present in the sample directory ([example](yolov3_onnx/download.yml)).
   ```bash
   downloader.py -d /path/to/data/dir -f /path/to/download.yml
   ```
