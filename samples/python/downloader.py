#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import errno
import hashlib
import logging
import os
import sys


logger = logging.getLogger("downloader")


class DataFile:
    """Holder of a data file."""

    def __init__(self, attr):
        self.attr = attr
        self.path = attr["path"]
        self.url = attr["url"]
        if "checksum" not in attr:
            logger.warning("Checksum of %s not provided!", self.path)
        self.checksum = attr.get("checksum", None)

    def __str__(self):
        return str(self.attr)


class SampleData:
    """Holder of data files of an sample."""

    def __init__(self, attr):
        self.attr = attr
        self.sample = attr["sample"]
        files = attr.get("files", None)
        self.files = [DataFile(f) for f in files]

    def __str__(self):
        return str(self.attr)


def _loadYAML(yaml_path):
    with open(yaml_path, "rb") as f:
        import yaml

        y = yaml.load(f, yaml.FullLoader)
        return SampleData(y)


def _checkMD5(path, refMD5):
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    return md5 == refMD5


def _createDirIfNeeded(path):
    the_dir = os.path.dirname(path)
    try:
        os.makedirs(the_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def download(data_dir, yaml_path, overwrite=False):
    """Download the data files specified in YAML file to a directory.

    Return false if the downloaded file or the local copy (if not overwrite) has a different checksum.
    """
    sample_data = _loadYAML(yaml_path)
    logger.info("Downloading data for %s", sample_data.sample)

    def _downloadFile(path, url):
        logger.info("Downloading %s from %s", path, url)
        import requests

        r = requests.get(url, stream=True, timeout=5)
        size = int(r.headers.get("content-length", 0))
        from tqdm import tqdm

        progress_bar = tqdm(total=size, unit="iB", unit_scale=True)
        with open(path, "wb") as fd:
            for chunk in r.iter_content(chunk_size=1024):
                progress_bar.update(len(chunk))
                fd.write(chunk)
        progress_bar.close()

    allGood = True
    for f in sample_data.files:
        fpath = os.path.join(data_dir, f.path)
        if os.path.exists(fpath):
            if _checkMD5(fpath, f.checksum):
                logger.info("Found local copy %s, skip downloading.", fpath)
                continue
            else:
                logger.warning("Local copy %s has a different checksum!", fpath)
                if overwrite:
                    logging.warning("Removing local copy %s", fpath)
                    os.remove(fpath)
                else:
                    allGood = False
                    continue
        _createDirIfNeeded(fpath)
        _downloadFile(fpath, f.url)
        if not _checkMD5(fpath, f.checksum):
            logger.error("The downloaded file %s has a different checksum!", fpath)
            allGood = False

    return allGood


def _parseArgs():
    parser = argparse.ArgumentParser(description="Downloader of TensorRT sample data files.")
    parser.add_argument(
        "-d",
        "--data",
        help="Specify the data directory, data will be downloaded to there. $TRT_DATA_DIR will be overwritten by this argument.",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Specify the path to the download.yml, default to `download.yml` in the working directory",
        default="download.yml",
    )
    parser.add_argument(
        "-o", "--overwrite", help="Force to overwrite if MD5 check failed", action="store_true", default=False
    )
    parser.add_argument(
        "-v",
        "--verify",
        help="Verify if the data has been downloaded. Will not download if specified.",
        action="store_true",
        default=False,
    )

    args, _ = parser.parse_known_args()
    data = os.environ.get("TRT_DATA_DIR", None) if args.data is None else args.data
    if data is None:
        raise ValueError("Data directory must be specified by either `-d $DATA` or environment variable $TRT_DATA_DIR.")

    return data, args


def verifyChecksum(data_dir, yaml_path):
    """Verify the checksum of the files described by the YAML.

    Return false of any of the file doesn't existed or checksum is different with the YAML.
    """
    sample_data = _loadYAML(yaml_path)
    logger.info("Verifying data files and their MD5 for %s", sample_data.sample)

    allGood = True
    for f in sample_data.files:
        fpath = os.path.join(data_dir, f.path)
        if os.path.exists(fpath):
            if _checkMD5(fpath, f.checksum):
                logger.info("MD5 match for local copy %s", fpath)
            else:
                logger.error("Local file %s has a different checksum!", fpath)
                allGood = False
        else:
            allGood = False
            logger.error("Data file %s doesn't have a local copy", f.path)

    return allGood


def main():
    data, args = _parseArgs()
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    ret = True
    if args.verify:
        ret = verifyChecksum(data, args.file)
    else:
        ret = download(data, args.file, args.overwrite)

    if not ret:
        # Error of downloading or checksum
        sys.exit(1)


if __name__ == "__main__":
    main()


TRT_DATA_DIR = None


def getFilePath(path):
    """Util to get the full path to the downloaded data files.

    It only works when the sample doesn't have any other command line argument.
    """
    global TRT_DATA_DIR
    if not TRT_DATA_DIR:
        parser = argparse.ArgumentParser(description="Helper of data file download tool")
        parser.add_argument(
            "-d",
            "--data",
            help="Specify the data directory where it is saved in. $TRT_DATA_DIR will be overwritten by this argument.",
        )
        args, _ = parser.parse_known_args()
        TRT_DATA_DIR = os.environ.get("TRT_DATA_DIR", None) if args.data is None else args.data
    if TRT_DATA_DIR is None:
        raise ValueError("Data directory must be specified by either `-d $DATA` or environment variable $TRT_DATA_DIR.")

    fullpath = os.path.join(TRT_DATA_DIR, path)
    if not os.path.exists(fullpath):
        raise ValueError("Data file %s doesn't exist!" % fullpath)

    return fullpath
