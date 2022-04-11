#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from PIL import Image
import urllib.request
import numpy as np
import argparse
import gzip
import os


# Returns a numpy buffer of shape (num_images, 28, 28)
def load_mnist_data(buffer):
    raw_buf = np.fromstring(buffer, dtype=np.uint8)
    # Make sure the magic number is what we expect
    assert raw_buf[0:4].view(">i4")[0] == 2051
    num_images = raw_buf[4:8].view(">i4")[0]
    image_h = raw_buf[8:12].view(">i4")[0]
    image_w = raw_buf[12:16].view(">i4")[0]
    # Colors in the dataset are inverted vs. what the samples expect.
    return np.ascontiguousarray(255 - raw_buf[16:].reshape(num_images, image_h, image_w))

# Returns a list of length num_images
def load_mnist_labels(buffer):
    raw_buf = np.fromstring(buffer, dtype=np.uint8)
    # Make sure the magic number is what we expect
    assert raw_buf[0:4].view(">i4")[0] == 2049
    num_labels = raw_buf[4:8].view(">i4")[0]
    return list(raw_buf[8:].astype(np.int32).reshape(num_labels))

def main():
    parser = argparse.ArgumentParser(description="Extracts 10 PGM files from the MNIST dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--output", help="Path to the output directory.", default=os.getcwd())

    args, _ = parser.parse_known_args()

    with urllib.request.urlopen("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz") as res:
        data = load_mnist_data(gzip.decompress(res.read()))

    with urllib.request.urlopen("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz") as res:
        labels = load_mnist_labels(gzip.decompress(res.read()))

    output_dir = args.output

    # Find one image for each digit.
    for i in range(10):
        index = labels.index(i)
        image = Image.fromarray(data[index], mode="L")
        path = os.path.join(output_dir, "{:}.pgm".format(i))
        image.save(path)

if __name__ == '__main__':
    main()

