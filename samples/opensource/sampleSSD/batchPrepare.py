#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import numpy as np
import sys
import os
import glob
import shutil
import struct
from random import shuffle

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Pillow installed.
For installation instructions, see:
http://pillow.readthedocs.io/en/stable/installation.html""".format(err))

height = 300
width = 300
NUM_BATCHES = 0
NUM_PER_BATCH = 1
NUM_CALIBRATION_IMAGES = 50

parser = argparse.ArgumentParser()
parser.add_argument('--inDir', required=True, help='Input directory')
parser.add_argument('--outDir', required=True, help='Output directory')

args = parser.parse_args()

CALIBRATION_DATASET_LOC = args.inDir + '/*.jpg'


# images to test
imgs = []
print("Location of dataset = " + CALIBRATION_DATASET_LOC)
imgs = glob.glob(CALIBRATION_DATASET_LOC)
shuffle(imgs)
imgs = imgs[:NUM_CALIBRATION_IMAGES]
NUM_BATCHES = NUM_CALIBRATION_IMAGES // NUM_PER_BATCH + (NUM_CALIBRATION_IMAGES % NUM_PER_BATCH > 0)

print("Total number of images = " + str(len(imgs)))
print("NUM_PER_BATCH = " + str(NUM_PER_BATCH))
print("NUM_BATCHES = " + str(NUM_BATCHES))

# output
outDir  = args.outDir+"/batches"

if os.path.exists(outDir):
	os.system("rm " + outDir +"/*")

# prepare output
if not os.path.exists(outDir):
	os.makedirs(outDir)

for i in range(NUM_CALIBRATION_IMAGES):
	os.system("convert "+imgs[i]+" -resize "+str(height)+"x"+str(width)+"! "+outDir+"/"+str(i)+".ppm")

CALIBRATION_DATASET_LOC= outDir + '/*.ppm'
imgs = glob.glob(CALIBRATION_DATASET_LOC)

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
img = 0
for i in range(NUM_BATCHES):
	batchfile = outDir + "/batch_calibration" + str(i) + ".batch"
	batchlistfile = outDir + "/batch_calibration" + str(i) + ".list"
	batchlist = open(batchlistfile,'a')
	batch = np.zeros(shape=(NUM_PER_BATCH, 3, height, width), dtype = np.float32)
	for j in range(NUM_PER_BATCH):
		batchlist.write(os.path.basename(imgs[img]) + '\n')
		im = Image.open(imgs[img]).resize((width,height), Image.NEAREST)
		in_ = np.array(im, dtype=np.float32, order='C')
		in_ = in_[:,:,::-1]
		in_-= np.array((104.0, 117.0, 123.0))
		in_ = in_.transpose((2,0,1))
		batch[j] = in_
		img += 1

	# save
	batch.tofile(batchfile)
	batchlist.close()

	# Prepend batch shape information
	ba = bytearray(struct.pack("4i", batch.shape[0], batch.shape[1], batch.shape[2], batch.shape[3]))

	with open(batchfile, 'rb+') as f:
		content = f.read()
		f.seek(0,0)
		f.write(ba)
		f.write(content)

os.system("rm " + outDir +"/*.ppm")
