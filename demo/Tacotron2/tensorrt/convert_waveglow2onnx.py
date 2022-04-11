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

import torch
import argparse
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from common.utils import ParseFromConfigFile
from inference import load_and_setup_model

def convert_convinv_1d_to_2d(convinv):
    """
    Takes an invertible 1x1 1-d convolution and returns a 2-d convolution that does
    the inverse
    """
    conv2d = torch.nn.Conv2d(convinv.W_inverse.size(1),
                             convinv.W_inverse.size(0),
                             1, bias=False)
    conv2d.weight.data[:,:,:,0] = convinv.W_inverse.data
    return conv2d


def convert_conv_1d_to_2d(conv1d):
    conv2d = torch.nn.Conv2d(conv1d.weight.size(1),
                             conv1d.weight.size(0),
                             (conv1d.weight.size(2), 1),
                             stride=(conv1d.stride[0], 1),
                             dilation=(conv1d.dilation[0], 1),
                             padding=(conv1d.padding[0], 0))
    conv2d.weight.data[:,:,:,0] = conv1d.weight.data
    conv2d.bias.data = conv1d.bias.data
    return conv2d


def convert_WN_1d_to_2d_(WN):
    """
    Modifies the WaveNet like affine coupling layer in-place to use 2-d convolutions
    """
    WN.start = convert_conv_1d_to_2d(WN.start)
    WN.end = convert_conv_1d_to_2d(WN.end)

    for i in range(len(WN.in_layers)):
        WN.in_layers[i] = convert_conv_1d_to_2d(WN.in_layers[i])

    for i in range(len(WN.res_skip_layers)):
        WN.res_skip_layers[i] = convert_conv_1d_to_2d(WN.res_skip_layers[i])

    for i in range(len(WN.res_skip_layers)):
        WN.cond_layers[i] = convert_conv_1d_to_2d(WN.cond_layers[i])


def convert_1d_to_2d_(glow):
    """
    Caffe2 and TensorRT don't seem to support 1-d convolutions or properly
    convert ONNX exports with 1d convolutions to 2d convolutions yet, so we
    do the conversion to 2-d convolutions before ONNX export
    """
    # Convert upsample to 2d
    upsample = torch.nn.ConvTranspose2d(glow.upsample.weight.size(0),
                                        glow.upsample.weight.size(1),
                                        (glow.upsample.weight.size(2), 1),
                                        stride=(glow.upsample.stride[0], 1))
    upsample.weight.data[:,:,:,0] = glow.upsample.weight.data
    upsample.bias.data = glow.upsample.bias.data
    glow.upsample = upsample.cuda()

    # Convert WN to 2d
    for WN in glow.WN:
        convert_WN_1d_to_2d_(WN)

    # Convert invertible conv to 2d
    for i in range(len(glow.convinv)):
        glow.convinv[i] = convert_convinv_1d_to_2d(glow.convinv[i])

    glow.cuda()

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--waveglow', type=str, required=True,
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory or file name for the exported WaveGlow ONNX model')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with AMP')
    parser.add_argument('-s', '--sigma-infer', default=0.6, type=float)

    parser.add_argument('--config-file', action=ParseFromConfigFile,
                         type=str, help='Path to configuration file')

    return parser


def export_onnx(parser, args):

    waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    fp16_run=args.fp16, cpu_run=False,
                                    forward_is_infer=False)

    # 80 mel channels, 620 mel spectrograms ~ 7 seconds of speech
    mel = torch.randn(1, 80, 620).cuda()
    stride = 256 # value from waveglow upsample
    n_group = 8
    z_size2 = (mel.size(2)*stride)//n_group
    z = torch.randn(1, n_group, z_size2, 1).cuda()

    if args.fp16:
        mel = mel.half()
        z = z.half()
    with torch.no_grad():
        # run inference to force calculation of inverses
        waveglow.infer(mel, sigma=args.sigma_infer)

        convert_1d_to_2d_(waveglow)
        mel = mel.unsqueeze(3)

        # export to ONNX
        if args.fp16:
            waveglow = waveglow.half()

        waveglow.forward = waveglow.infer_onnx

        opset_version = 11

        if os.path.isdir(args.output):
            output_path = os.path.join(args.output, "waveglow.onnx")
        else:
            output_path = args.output

        torch.onnx.export(waveglow, (mel, z), output_path,
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=["mel", "z"],
                          output_names=["audio"],
                          dynamic_axes={"mel":   {0: "batch_size", 2: "mel_seq"},
                                        "z":     {0: "batch_size", 2: "z_seq"},
                                        "audio": {0: "batch_size", 1: "audio_seq"}})


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    export_onnx(parser, args)

if __name__ == '__main__':
    main()
