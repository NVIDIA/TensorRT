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

"""
Obtain the benchmark timing and output from the original HuggingFace BART model.

Usage: python3 hf.py --variant facebook/bart-base [--enable-kv-cache] [--fp16]
"""

import time 
from transformers import BartTokenizer, BartForConditionalGeneration
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--variant", help="Name of BART variant.")
parser.add_argument("--enable-kv-cache", help="Bart enable KV cache", action="store_true", default=False)
parser.add_argument("--fp16", help="Bart FP16", action="store_true", default=False)
parser.add_argument("--num-beams", type=int, default=1, help="Enables beam search during decoding.")

args = parser.parse_args()

model = BartForConditionalGeneration.from_pretrained(args.variant) # facebook/bart-base, facebook/bart-large, facebook/bart-large-cnn
tokenizer = BartTokenizer.from_pretrained(args.variant)
model = model.to('cuda').eval()

if args.fp16:
    model = model.half()

ARTICLE_TO_SUMMARIZE = (
    "NVIDIA TensorRT-based applications perform up to 36X faster than CPU-only platforms during inference, enabling developers to optimize neural network models trained on all major frameworks, calibrate for lower precision with high accuracy, and deploy to hyperscale data centers, embedded platforms, or automotive product platforms. TensorRT, built on the NVIDIA CUDA parallel programming model, enables developers to optimize inference by leveraging libraries, development tools, and technologies in CUDA-X for AI, autonomous machines, high performance computing, and graphics. With new NVIDIA Ampere Architecture GPUs, TensorRT also uses sparse tensor cores for an additional performance boost."
)

input_ids = tokenizer([ARTICLE_TO_SUMMARIZE], padding=True, return_tensors="pt").input_ids.to('cuda')

warmup = 10
for i in range(warmup):
    summary_ids = model.generate(input_ids, max_length=1024, num_beams=args.num_beams, use_cache=args.enable_kv_cache)

start = time.time()
trials = 10

input_ids = tokenizer([ARTICLE_TO_SUMMARIZE], padding=True, return_tensors="pt").input_ids.to('cuda')

for i in range(trials):
    # Generate Summary. Note: generate() method already has torch.no_grad() decorator.
    summary_ids = model.generate(input_ids, max_length=1024, num_beams=args.num_beams, use_cache=args.enable_kv_cache)

end = time.time()

output = tokenizer.decode(summary_ids[-1,:], skip_special_tokens=True)

print('BART output: ', output) 
print(f"Input sequence length: {input_ids.size(1)}, Output sequence length: {summary_ids[-1,:].size(0)}")
print("Average run time: {:.2f} ms".format((end - start)/trials*1000))
