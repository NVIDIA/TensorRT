General Setup Guide for Samples
==============================


## Download Sample Data

Install the tool dependencies via `python3 -m pip install -r requirements.txt`.

Invoke [downloader.py](downloader.py) to download the data with
a command like the one below if `download.yml` is present in the
sample directory ([example](yolov3_onnx/download.yml)).

```sh
downloader.py -d /path/to/data/dir -f /path/to/download.yml
```

The data directory i.e. `/path/to/data/dir` is a centralized directory
to store data of all samples. So you can use same one for all samples.
It can be provided by either `-d /path/to/data/dir` or the environment variable
`$TRT_DATA_DIR`, where the `-d` has higher priority.

Remember to use `-d` or `$TRT_DATA_DIR` when running sample scripts
that rely on downloaded data. Scripts will abort if no downloaded data
is found in data directory. (`$TRT_DATA_DIR` will be much simplier.)
An error will be thrown if the data is not properly setup.

The `download.yml` file is owned by the sample which describes the sample
name, the path, URL and checksum of the data files that are required by the sample.


**Notes for sample developers**

To use the downloaded data files, integrate the code segment like below into
the sample code, and obtain the path to the data file by passing the `path`
as specified in the associated `download.yml` file of the sample.

```py
TRT_DATA_DIR = None

def getFilePath(path):
    global TRT_DATA_DIR
    if not TRT_DATA_DIR:
        parser = argparse.ArgumentParser(description="Convert YOLOv3 to ONNX model")
        parser.add_argument('-d', '--data', help="Specify the data directory where it is saved in. $TRT_DATA_DIR will be overwritten by this argument.")
        args, _ = parser.parse_known_args()
        TRT_DATA_DIR = os.environ.get('TRT_DATA_DIR', None) if args.data is None else args.data
    if TRT_DATA_DIR is None:
        raise ValueError("Data directory must be specified by either `-d $DATA` or environment variable $TRT_DATA_DIR.")

    fullpath = os.path.join(TRT_DATA_DIR, path)
    if not os.path.exists(fullpath):
        raise ValueError("Data file %s doesn't exist!" % fullpath)

    return fullpath
```

The helper function `getFilePath` in `downloader.py` can also be used to obtain the full path to the downloaded data files. It only works when the sample doesn't have any other command line argument.

```py
from downloader import getFilePath

cfg_file_path = getFilePath('samples/python/yolov3_onnx/yolov3.cfg')
```
