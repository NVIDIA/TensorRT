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

import argparse
import onnx
import pycuda.autoinit
import pycuda.driver as cuda
import sys
import tensorrt as trt
from os.path import join

from trt_utils import build_engine, parse_dynamic_size

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--encoder', type=str, default="",
                        help='full path to the Encoder ONNX')
    parser.add_argument('--decoder', type=str, default="",
                        help='full path to the Decoder or DecoderIter ONNX.')
    parser.add_argument('--postnet', type=str, default="",
                        help='full path to the Postnet ONNX')
    parser.add_argument('--waveglow', type=str, default="",
                        help='full path to the WaveGlow ONNX')
    parser.add_argument('--encoder_out', type=str,
                        help='Filename of the exported encoder engine')
    parser.add_argument('--decoder_out', type=str,
                        help='Filename of the exported decoder engine')
    parser.add_argument('--postnet_out', type=str,
                        help='Filename of the exported postnet engine')
    parser.add_argument('--waveglow_out', type=str,
                        help='Filename of the exported waveglow engine')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with FP16')
    parser.add_argument('-bs', '--batch-size', type=str, default="1",
                        help='One or three comma separated integers specifying the batch size. Specify "min,opt,max" for dynamic shape')
    parser.add_argument('--mel-size', type=str, default="32,768,1664",
                        help='One or three comma separated integers specifying the mels size for waveglow.')
    parser.add_argument('--z-size', type=str, default="1024,24576,53248",
                        help='One or three comma separated integers specifying the z size for waveglow.')
    parser.add_argument('--loop', dest='loop', action='store_true',
                        help='Includes the outer decoder loop in the ONNX model. Enabled by default and only supported on TensorRT 8.0 or later.')
    parser.add_argument('--no-loop', dest='loop', action='store_false',
                        help='Excludes outer decoder loop from decoder ONNX model. Default behavior and necessary for TensorRT 7.2 or earlier.')
    parser.add_argument("-tcf", "--timing-cache-file", default=None, type=str,
                        help="Path to tensorrt build timeing cache file, only available for tensorrt 8.0 and later. The cache file is assumed to be used exclusively. It's the users' responsibility to create file lock to prevent accessing conflict.",
                        required=False)
    parser.add_argument("--faster-dynamic-shapes", action="store_true", help="Enable dynamic shape preview feature.")
    parser.set_defaults(loop=int(trt.__version__[0]) >= 8)

    return parser


def main():

    parser = argparse.ArgumentParser(
        description='Export from ONNX to TensorRT for Tacotron 2 and WaveGlow')
    parser = parse_args(parser)
    args = parser.parse_args()

    precision = "fp16" if args.fp16 else "fp32"
    encoder_path = join(args.output, args.encoder_out if args.encoder_out else f"encoder_{precision}.engine")
    decoder_path = join(args.output, args.decoder_out if args.decoder_out else f"decoder_with_outer_loop_{precision}.engine" if args.loop else f"decoder_iter_{precision}.engine")
    postnet_path = join(args.output, args.postnet_out if args.postnet_out else f"postnet_{precision}.engine")
    waveglow_path = join(args.output, args.waveglow_out if args.waveglow_out else f"waveglow_{precision}.engine")

    bs_min, bs_opt, bs_max = parse_dynamic_size(args.batch_size)
    mel_min, mel_opt, mel_max = parse_dynamic_size(args.mel_size)
    z_min, z_opt, z_max = parse_dynamic_size(args.z_size)

    # Encoder
    shapes=[{"name": "sequences",        "min": (bs_min,4), "opt": (bs_opt,128), "max": (bs_max,256)},
            {"name": "sequence_lengths", "min": (bs_min,),  "opt": (bs_opt,),    "max": (bs_max,)}]
    if args.encoder != "":
        print("Building Encoder ...")
        encoder_engine = build_engine(args.encoder, shapes=shapes, fp16=args.fp16, timing_cache=args.timing_cache_file, faster_dynamic_shapes=args.faster_dynamic_shapes)
        if encoder_engine is not None:
            with open(encoder_path, 'wb') as f:
                f.write(encoder_engine.serialize())
        else:
            print("Failed to build engine from", args.encoder)
            sys.exit(1)

    if args.loop:
        # Decoder
        shapes=[{"name": "decoder_input_0",         "min": (bs_min,80),    "opt": (bs_opt,80),      "max": (bs_max,80)},
                {"name": "attention_hidden_0",      "min": (bs_min,1024),  "opt": (bs_opt,1024),    "max": (bs_max,1024)},
                {"name": "attention_cell_0",        "min": (bs_min,1024),  "opt": (bs_opt,1024),    "max": (bs_max,1024)},
                {"name": "decoder_hidden_0",        "min": (bs_min,1024),  "opt": (bs_opt,1024),    "max": (bs_max,1024)},
                {"name": "decoder_cell_0",          "min": (bs_min,1024),  "opt": (bs_opt,1024),    "max": (bs_max,1024)},
                {"name": "attention_weights_0",     "min": (bs_min,4),     "opt": (bs_opt,128),     "max": (bs_max,256)},
                {"name": "attention_weights_cum_0", "min": (bs_min,4),     "opt": (bs_opt,128),     "max": (bs_max,256)},
                {"name": "attention_context_0",     "min": (bs_min,512),   "opt": (bs_opt,512),     "max": (bs_max,512)},
                {"name": "memory",                  "min": (bs_min,4,512), "opt": (bs_opt,128,512), "max": (bs_max,256,512)},
                {"name": "processed_memory",        "min": (bs_min,4,128), "opt": (bs_opt,128,128), "max": (bs_max,256,128)},
                {"name": "mask",                    "min": (bs_min,4),     "opt": (bs_opt,128),     "max": (bs_max,256)}]
        if args.decoder != "":
            print("Building Decoder with loop...")
            decoder_engine = build_engine(args.decoder, shapes=shapes, fp16=args.fp16, timing_cache=args.timing_cache_file, faster_dynamic_shapes=args.faster_dynamic_shapes)
            if decoder_engine is not None:
                with open(decoder_path, 'wb') as f:
                    f.write(decoder_engine.serialize())
            else:
                print("Failed to build engine from", args.decoder)
                sys.exit(1)
    else:
        # DecoderIter
        shapes=[{"name": "decoder_input",         "min": (bs_min,80),    "opt": (bs_opt,80),      "max": (bs_max,80)},
                {"name": "attention_hidden",      "min": (bs_min,1024),  "opt": (bs_opt,1024),    "max": (bs_max,1024)},
                {"name": "attention_cell",        "min": (bs_min,1024),  "opt": (bs_opt,1024),    "max": (bs_max,1024)},
                {"name": "decoder_hidden",        "min": (bs_min,1024),  "opt": (bs_opt,1024),    "max": (bs_max,1024)},
                {"name": "decoder_cell",          "min": (bs_min,1024),  "opt": (bs_opt,1024),    "max": (bs_max,1024)},
                {"name": "attention_weights",     "min": (bs_min,4),     "opt": (bs_opt,128),     "max": (bs_max,256)},
                {"name": "attention_weights_cum", "min": (bs_min,4),     "opt": (bs_opt,128),     "max": (bs_max,256)},
                {"name": "attention_context",     "min": (bs_min,512),   "opt": (bs_opt,512),     "max": (bs_max,512)},
                {"name": "memory",                "min": (bs_min,4,512), "opt": (bs_opt,128,512), "max": (bs_max,256,512)},
                {"name": "processed_memory",      "min": (bs_min,4,128), "opt": (bs_opt,128,128), "max": (bs_max,256,128)},
                {"name": "mask",                  "min": (bs_min,4),     "opt": (bs_opt,128),     "max": (bs_max,256)}]
        if args.decoder != "":
            print("Building Decoder ...")
            decoder_iter_engine = build_engine(args.decoder, shapes=shapes, fp16=args.fp16, timing_cache=args.timing_cache_file, faster_dynamic_shapes=args.faster_dynamic_shapes)
            if decoder_iter_engine is not None:
                with open(decoder_path, 'wb') as f:
                    f.write(decoder_iter_engine.serialize())
            else:
                print("Failed to build engine from", args.decoder)
                sys.exit(1)

    # Postnet
    shapes=[{"name": "mel_outputs", "min": (bs_min,80,32), "opt": (bs_opt,80,768), "max": (bs_max,80,1664)}]
    if args.postnet != "":
        print("Building Postnet ...")
        postnet_engine = build_engine(args.postnet, shapes=shapes, fp16=args.fp16, timing_cache=args.timing_cache_file, faster_dynamic_shapes=args.faster_dynamic_shapes)
        if postnet_engine is not None:
            with open(postnet_path, 'wb') as f:
                f.write(postnet_engine.serialize())
        else:
            print("Failed to build engine from", args.postnet)
            sys.exit(1)

    # WaveGlow
    shapes=[{"name": "mel", "min": (bs_min,80,mel_min,1),  "opt": (bs_opt,80,mel_opt,1),  "max": (bs_max,80,mel_max,1)},
            {"name": "z",   "min": (bs_min,8,z_min,1),     "opt": (bs_opt,8,z_opt,1),     "max": (bs_max,8,z_max,1)}]
    if args.waveglow != "":
        print("Building WaveGlow ...")
        waveglow_engine = build_engine(args.waveglow, shapes=shapes, fp16=args.fp16, timing_cache=args.timing_cache_file, faster_dynamic_shapes=args.faster_dynamic_shapes)
        if waveglow_engine is not None:
            with open(waveglow_path, 'wb') as f:
                f.write(waveglow_engine.serialize())
        else:
            print("Failed to build engine from", args.waveglow)
            sys.exit(1)


if __name__ == '__main__':
    main()
