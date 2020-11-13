#!/bin/bash
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

echo "TensorRT BS=1, S=128"
bash test_infer.sh --test trt/test_infer_trt.py -bs 1 -il 128 --fp16 --num-iters 1003 --encoder ./output/encoder_fp16.engine --decoder ./output/decoder_iter_fp16.engine --postnet ./output/postnet_fp16.engine --waveglow ./output/waveglow_fp16.engine --wn-channels 256
echo "PyTorch (GPU) BS=1, S=128"
bash test_infer.sh -bs 1 -il 128 --fp16 --num-iters 1003 --tacotron2 ./checkpoints/tacotron2_pyt_ckpt_amp_v19.09.0/nvidia_tacotron2pyt_fp16_20190427 --waveglow ./checkpoints/waveglow_ckpt_amp_256_v19.10.0/nvidia_waveglow256pyt_fp16 --wn-channels 256
