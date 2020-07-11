# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import onnx
import argparse

import sys
sys.path.append('./')

from trt.trt_utils import build_engine

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--encoder', type=str, default="",
                        help='full path to the Encoder ONNX')
    parser.add_argument('--decoder', type=str, default="",
                        help='full path to the DecoderIter ONNX')
    parser.add_argument('--postnet', type=str, default="",
                        help='full path to the Postnet ONNX')
    parser.add_argument('--waveglow', type=str, default="",
                        help='full path to the WaveGlow ONNX')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with FP16')

    return parser


def main():

    parser = argparse.ArgumentParser(
        description='Export from ONNX to TensorRT for Tacotron 2 and WaveGlow')
    parser = parse_args(parser)
    args = parser.parse_args()

    engine_prec = "_fp16" if args.fp16 else "_fp32"

    # Encoder
    shapes=[{"name": "sequences",        "min": (1,4), "opt": (1,128), "max": (1,256)},
            {"name": "sequence_lengths", "min": (1,),  "opt": (1,),    "max": (1,)}]
    if args.encoder != "":
        print("Building Encoder ...")
        encoder_engine = build_engine(args.encoder, shapes=shapes, fp16=args.fp16)
        if encoder_engine is not None:
            with open(args.output+"/"+"encoder"+engine_prec+".engine", 'wb') as f:
                f.write(encoder_engine.serialize())
        else:
            print("Failed to build engine from", args.encoder)
            sys.exit()

    # DecoderIter
    shapes=[{"name": "decoder_input",         "min": (1,80),    "opt": (1,80),      "max": (1,80)},
            {"name": "attention_hidden",      "min": (1,1024),  "opt": (1,1024),    "max": (1,1024)},
            {"name": "attention_cell",        "min": (1,1024),  "opt": (1,1024),    "max": (1,1024)},
            {"name": "decoder_hidden",        "min": (1,1024),  "opt": (1,1024),    "max": (1,1024)},
            {"name": "decoder_cell",          "min": (1,1024),  "opt": (1,1024),    "max": (1,1024)},
            {"name": "attention_weights",     "min": (1,4),     "opt": (1,128),     "max": (1,256)},
            {"name": "attention_weights_cum", "min": (1,4),     "opt": (1,128),     "max": (1,256)},
            {"name": "attention_context",     "min": (1,512),   "opt": (1,512),     "max": (1,512)},
            {"name": "memory",                "min": (1,4,512), "opt": (1,128,512), "max": (1,256,512)},
            {"name": "processed_memory",      "min": (1,4,128), "opt": (1,128,128), "max": (1,256,128)},
            {"name": "mask",                  "min": (1,4),     "opt": (1,128),     "max": (1,256)}]
    if args.decoder != "":
        print("Building Decoder ...")
        decoder_iter_engine = build_engine(args.decoder, shapes=shapes, fp16=args.fp16)
        if decoder_iter_engine is not None:
            with open(args.output+"/"+"decoder_iter"+engine_prec+".engine", 'wb') as f:
                f.write(decoder_iter_engine.serialize())
        else:
            print("Failed to build engine from", args.decoder)
            sys.exit()

    # Postnet
    shapes=[{"name": "mel_outputs", "min": (1,80,32), "opt": (1,80,768), "max": (1,80,1664)}]
    if args.postnet != "":
        print("Building Postnet ...")
        postnet_engine = build_engine(args.postnet, shapes=shapes, fp16=args.fp16)
        if postnet_engine is not None:
            with open(args.output+"/"+"postnet"+engine_prec+".engine", 'wb') as f:
                f.write(postnet_engine.serialize())
        else:
            print("Failed to build engine from", args.postnet)
            sys.exit()

    # WaveGlow
    shapes=[{"name": "mel", "min": (1,80,32),  "opt": (1,80,768),  "max": (1,80,1664)},
            {"name": "z",   "min": (1,8,1024), "opt": (1,8,24576), "max": (1,8,53248)}]
    if args.waveglow != "":
        print("Building WaveGlow ...")
        waveglow_engine = build_engine(args.waveglow, shapes=shapes, fp16=args.fp16)
        if waveglow_engine is not None:
            with open(args.output+"/"+"waveglow"+engine_prec+".engine", 'wb') as f:
                f.write(waveglow_engine.serialize())
        else:
            print("Failed to build engine from", args.waveglow)
            sys.exit()


if __name__ == '__main__':
    main()
