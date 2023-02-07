#!/bin/bash
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

pip3 install --force-reinstall torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

echo "TensorRT BS=1, S=128"
bash test_infer.sh --test tensorrt/test_infer_trt.py -bs 1 -il 128 --fp16 --num-iters 103 --encoder ./output/encoder_fp16.engine --decoder ./output/decoder_with_outer_loop_fp16.engine --postnet ./output/postnet_fp16.engine --waveglow ./output/waveglow_fp16.engine --wn-channels 256
echo "PyTorch (GPU) BS=1, S=128"
bash test_infer.sh -bs 1 -il 128 --fp16 --num-iters 103 --tacotron2 ./checkpoints/tacotron2_pyt_ckpt_amp_v19.09.0/nvidia_tacotron2pyt_fp16_20190427 --waveglow ./checkpoints/waveglow_ckpt_amp_256_v19.10.0/nvidia_waveglow256pyt_fp16 --wn-channels 256

pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

echo "PyTorch (CPU) BS=1, S=128"
bash test_infer.sh -bs 1 -il 128 --fp16 --num-iters 5 --tacotron2 ./checkpoints/tacotron2_pyt_ckpt_amp_v19.09.0/nvidia_tacotron2pyt_fp16_20190427 --waveglow ./checkpoints/waveglow_ckpt_amp_256_v19.10.0/nvidia_waveglow256pyt_fp16 --wn-channels 256 --cpu
