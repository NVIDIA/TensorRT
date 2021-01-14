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

unset CUDA_VISIBLE_DEVICES
bash test_infer.sh -bs 1 -il 128 --fp16 --num-iters 1003 --tacotron2 ./checkpoints/tacotron2_1032590_6000_amp --waveglow ./checkpoints/waveglow_1076430_14000_amp --wn-channels 256
bash test_infer.sh -bs 4 -il 128 --fp16 --num-iters 1003 --tacotron2 ./checkpoints/tacotron2_1032590_6000_amp --waveglow ./checkpoints/waveglow_1076430_14000_amp --wn-channels 256
bash test_infer.sh -bs 1 -il 128 --num-iters 1003 --tacotron2 ./checkpoints/tacotron2_1032590_6000_amp --waveglow ./checkpoints/waveglow_1076430_14000_amp --wn-channels 256
bash test_infer.sh -bs 4 -il 128 --num-iters 1003 --tacotron2 ./checkpoints/tacotron2_1032590_6000_amp --waveglow ./checkpoints/waveglow_1076430_14000_amp --wn-channels 256
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=6
export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,compact,1,0
bash test_infer.sh -bs 1 -il 128 --cpu --num-iters 1003 --tacotron2 ./checkpoints/tacotron2_1032590_6000_amp --waveglow ./checkpoints/waveglow_1076430_14000_amp --wn-channels 256
bash test_infer.sh -bs 4 -il 128 --cpu --num-iters 1003 --tacotron2 ./checkpoints/tacotron2_1032590_6000_amp --waveglow ./checkpoints/waveglow_1076430_14000_amp --wn-channels 256
