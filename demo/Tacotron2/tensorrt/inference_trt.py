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

import tensorrt as trt
import numpy as np
from scipy.io.wavfile import write
import time
import torch
import argparse
import os.path as path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from common.utils import to_gpu, get_mask_from_lengths
from tacotron2.text import text_to_sequence
from inference import MeasureTime, prepare_input_sequence, load_and_setup_model
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from trt_utils import load_engine, run_trt_engine

from waveglow.denoiser import Denoiser

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--encoder', type=str, required=True,
                        help='full path to the Encoder engine')
    parser.add_argument('--decoder', type=str, required=True,
                        help='full path to the DecoderIter engine')
    parser.add_argument('--postnet', type=str, required=True,
                        help='full path to the Postnet engine')
    parser.add_argument('--waveglow', type=str, required=True,
                        help='full path to the WaveGlow engine')
    parser.add_argument('--waveglow-ckpt', type=str, default="",
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--fp16', action='store_true',
                        help='inference with FP16')
    parser.add_argument('--loop', dest='loop', action='store_true',
                        help='Includes the outer decoder loop in the ONNX model. Enabled by default and only supported on TensorRT 8.0 or later.')
    parser.add_argument('--no-loop', dest='loop', action='store_false',
                        help='Excludes outer decoder loop from decoder ONNX model. Default behavior and necessary for TensorRT 7.2 or earlier.')
    parser.set_defaults(loop=int(trt.__version__[0]) >= 8)
    parser.add_argument('--waveglow-onnxruntime', action='store_true',
                        help='Specify this option to use ONNX runtime instead of TRT for running Waveglow')
    parser.add_argument('--decoder-onnxruntime', action='store_true',
                        help='Specify this option to use ONNX runtime instead of TRT for running the TT2 Decoder with loop. When using this option, pass the decoder ONNX model to the --decoder argument')
    return parser


def init_decoder_inputs(memory, processed_memory, memory_lengths):

    device = memory.device
    dtype = memory.dtype
    bs = memory.size(0)
    seq_len = memory.size(1)
    attention_rnn_dim = 1024
    decoder_rnn_dim = 1024
    encoder_embedding_dim = 512
    n_mel_channels = 80

    attention_hidden = torch.zeros(bs, attention_rnn_dim, device=device, dtype=dtype)
    attention_cell = torch.zeros(bs, attention_rnn_dim, device=device, dtype=dtype)
    decoder_hidden = torch.zeros(bs, decoder_rnn_dim, device=device, dtype=dtype)
    decoder_cell = torch.zeros(bs, decoder_rnn_dim, device=device, dtype=dtype)
    attention_weights = torch.zeros(bs, seq_len, device=device, dtype=dtype)
    attention_weights_cum = torch.zeros(bs, seq_len, device=device, dtype=dtype)
    attention_context = torch.zeros(bs, encoder_embedding_dim, device=device, dtype=dtype)
    mask = get_mask_from_lengths(memory_lengths).to(device)
    decoder_input = torch.zeros(bs, n_mel_channels, device=device, dtype=dtype)

    return (decoder_input, attention_hidden, attention_cell, decoder_hidden,
            decoder_cell, attention_weights, attention_weights_cum,
            attention_context, memory, processed_memory, mask)

def init_decoder_outputs(memory, memory_lengths):

    device = memory.device
    dtype = memory.dtype
    bs = memory.size(0)
    seq_len = memory.size(1)
    attention_rnn_dim = 1024
    decoder_rnn_dim = 1024
    encoder_embedding_dim = 512
    n_mel_channels = 80

    attention_hidden = torch.zeros(bs, attention_rnn_dim, device=device, dtype=dtype)
    attention_cell = torch.zeros(bs, attention_rnn_dim, device=device, dtype=dtype)
    decoder_hidden = torch.zeros(bs, decoder_rnn_dim, device=device, dtype=dtype)
    decoder_cell = torch.zeros(bs, decoder_rnn_dim, device=device, dtype=dtype)
    attention_weights = torch.zeros(bs, seq_len, device=device, dtype=dtype)
    attention_weights_cum = torch.zeros(bs, seq_len, device=device, dtype=dtype)
    attention_context = torch.zeros(bs, encoder_embedding_dim, device=device, dtype=dtype)
    decoder_output = torch.zeros(bs, n_mel_channels, device=device, dtype=dtype)
    gate_prediction = torch.zeros(bs, 1, device=device, dtype=dtype)

    return (attention_hidden, attention_cell, decoder_hidden,
            decoder_cell, attention_weights, attention_weights_cum,
            attention_context, decoder_output, gate_prediction)

def init_decoder_tensors(decoder_inputs, decoder_outputs):

    decoder_tensors = {
        "inputs" : {
            'decoder_input': decoder_inputs[0],
            'attention_hidden': decoder_inputs[1],
            'attention_cell': decoder_inputs[2],
            'decoder_hidden': decoder_inputs[3],
            'decoder_cell': decoder_inputs[4],
            'attention_weights': decoder_inputs[5],
            'attention_weights_cum': decoder_inputs[6],
            'attention_context': decoder_inputs[7],
            'memory': decoder_inputs[8],
            'processed_memory': decoder_inputs[9],
            'mask': decoder_inputs[10]
        },
        "outputs" : {
            'out_attention_hidden': decoder_outputs[0],
            'out_attention_cell': decoder_outputs[1],
            'out_decoder_hidden': decoder_outputs[2],
            'out_decoder_cell': decoder_outputs[3],
            'out_attention_weights': decoder_outputs[4],
            'out_attention_weights_cum': decoder_outputs[5],
            'out_attention_context': decoder_outputs[6],
            'decoder_output': decoder_outputs[7],
            'gate_prediction': decoder_outputs[8]
        }
    }
    return decoder_tensors

def swap_inputs_outputs(decoder_inputs, decoder_outputs):

    new_decoder_inputs = (decoder_outputs[7], # decoder_output
                          decoder_outputs[0], # attention_hidden
                          decoder_outputs[1], # attention_cell
                          decoder_outputs[2], # decoder_hidden
                          decoder_outputs[3], # decoder_cell
                          decoder_outputs[4], # attention_weights
                          decoder_outputs[5], # attention_weights_cum
                          decoder_outputs[6], # attention_context
                          decoder_inputs[8],  # memory
                          decoder_inputs[9],  # processed_memory
                          decoder_inputs[10]) # mask

    new_decoder_outputs = (decoder_inputs[1], # attention_hidden
                           decoder_inputs[2], # attention_cell
                           decoder_inputs[3], # decoder_hidden
                           decoder_inputs[4], # decoder_cell
                           decoder_inputs[5], # attention_weights
                           decoder_inputs[6], # attention_weights_cum
                           decoder_inputs[7], # attention_context
                           decoder_inputs[0], # decoder_input
                           decoder_outputs[8])# gate_output

    return new_decoder_inputs, new_decoder_outputs


def infer_tacotron2_trt(encoder, decoder_iter, postnet,
                        encoder_context, decoder_context, postnet_context,
                        sequences, sequence_lengths, measurements, fp16, loop):

    batch_size = len(sequence_lengths)
    max_sequence_len = sequence_lengths[0]
    memory = torch.zeros((batch_size, max_sequence_len, 512)).cuda()
    if fp16:
        memory = memory.half()
    device = memory.device
    dtype = memory.dtype

    processed_memory = torch.zeros((batch_size, max_sequence_len, 128), device=device, dtype=dtype)
    lens = torch.zeros_like(sequence_lengths)
    print(f"batch_size: {batch_size}, max sequence length: {max_sequence_len}")

    encoder_tensors = {
        "inputs" :
        {'sequences': sequences, 'sequence_lengths': sequence_lengths},
        "outputs" :
        {'memory': memory, 'lens': lens, 'processed_memory': processed_memory}
    }

    print("Running Tacotron2 Encoder")
    with MeasureTime(measurements, "tacotron2_encoder_time"):
        run_trt_engine(encoder_context, encoder, encoder_tensors)
    max_decoder_steps = 1024
    device = memory.device
    mel_lengths = torch.zeros([memory.size(0)], dtype=torch.int32, device = device)
    not_finished = torch.ones([memory.size(0)], dtype=torch.int32, device = device)
    mel_outputs = torch.ones((batch_size, 80, max_decoder_steps), device = device, dtype=dtype).cuda()
    gate_threshold = 0.5
    first_iter = True

    decoder_inputs = init_decoder_inputs(memory, processed_memory, sequence_lengths)
    decoder_outputs = init_decoder_outputs(memory, sequence_lengths)

    if loop:
        if decoder_context is None:
            print("Running Tacotron2 Decoder with loop with ONNX-RT")
            decoder_inputs_onnxrt = [x.cpu().numpy().copy() for x in decoder_inputs]
            import onnx
            import onnxruntime
            sess = onnxruntime.InferenceSession(decoder_iter)

            with MeasureTime(measurements, "tacotron2_decoder_time"):
                result = sess.run(["mel_outputs", "mel_lengths_t"], {
                    'decoder_input_0': decoder_inputs_onnxrt[0],
                        'attention_hidden_0': decoder_inputs_onnxrt[1],
                        'attention_cell_0': decoder_inputs_onnxrt[2],
                        'decoder_hidden_0': decoder_inputs_onnxrt[3],
                        'decoder_cell_0': decoder_inputs_onnxrt[4],
                        'attention_weights_0': decoder_inputs_onnxrt[5],
                        'attention_weights_cum_0': decoder_inputs_onnxrt[6],
                        'attention_context_0': decoder_inputs_onnxrt[7],
                        'memory': decoder_inputs_onnxrt[8],
                        'processed_memory': decoder_inputs_onnxrt[9],
                        'mask': decoder_inputs_onnxrt[10]
                })

            mel_outputs = torch.tensor(result[0], device=device)
            mel_lengths = torch.tensor(result[1], device=device)
        else: 
            print("Running Tacotron2 Decoder with loop")
            decoder_tensors = {
                "inputs" :
                {
                    'decoder_input_0': decoder_inputs[0],
                    'attention_hidden_0': decoder_inputs[1],
                    'attention_cell_0': decoder_inputs[2],
                    'decoder_hidden_0': decoder_inputs[3],
                    'decoder_cell_0': decoder_inputs[4],
                    'attention_weights_0': decoder_inputs[5],
                    'attention_weights_cum_0': decoder_inputs[6],
                    'attention_context_0': decoder_inputs[7],
                    'memory': decoder_inputs[8],
                    'processed_memory': decoder_inputs[9],
                    'mask': decoder_inputs[10]
                },
                "outputs" :
                {'mel_outputs': mel_outputs, 'mel_lengths_t': mel_lengths}
            }

            with MeasureTime(measurements, "tacotron2_decoder_time"):
                run_trt_engine(decoder_context, decoder_iter, decoder_tensors)
            mel_outputs = mel_outputs[:,:,:torch.max(mel_lengths)]

    else:
        print("Running Tacotron2 Decoder")
        measurements_decoder = {}
        while True:
            decoder_tensors = init_decoder_tensors(decoder_inputs, decoder_outputs)
            with MeasureTime(measurements_decoder, "step"):
                run_trt_engine(decoder_context, decoder_iter, decoder_tensors)

            if first_iter:
                mel_outputs = torch.unsqueeze(decoder_outputs[7], 2)
                gate_outputs = torch.unsqueeze(decoder_outputs[8], 2)
                alignments = torch.unsqueeze(decoder_outputs[4], 2)
                measurements['tacotron2_decoder_time'] = measurements_decoder['step']
                first_iter = False
            else:
                mel_outputs = torch.cat((mel_outputs, torch.unsqueeze(decoder_outputs[7], 2)), 2)
                gate_outputs = torch.cat((gate_outputs, torch.unsqueeze(decoder_outputs[8], 2)), 2)
                alignments = torch.cat((alignments, torch.unsqueeze(decoder_outputs[4], 2)), 2)
                measurements['tacotron2_decoder_time'] += measurements_decoder['step']

            dec = torch.le(torch.sigmoid(decoder_outputs[8]), gate_threshold).to(torch.int32).squeeze(1)
            not_finished = not_finished*dec
            mel_lengths += not_finished

            if torch.sum(not_finished) == 0:
                print("Stopping after",mel_outputs.size(2),"decoder steps")
                break
            if mel_outputs.size(2) == max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_inputs, decoder_outputs = swap_inputs_outputs(decoder_inputs, decoder_outputs)

    mel_outputs = mel_outputs.clone().detach()
    mel_outputs_postnet = torch.zeros_like(mel_outputs, device=device, dtype=dtype)

    postnet_tensors = {
        "inputs" :
        {'mel_outputs': mel_outputs},
        "outputs" :
        {'mel_outputs_postnet': mel_outputs_postnet}
    }
    print("Running Tacotron2 Postnet")
    with MeasureTime(measurements, "tacotron2_postnet_time"):
        run_trt_engine(postnet_context, postnet, postnet_tensors)

    print("Tacotron2 Postnet done")

    return mel_outputs_postnet, mel_lengths


def infer_waveglow_trt(waveglow, waveglow_context, mel, measurements, fp16):

    mel_size = mel.size(2)
    batch_size = mel.size(0)
    stride = 256
    n_group = 8
    z_size = mel_size*stride
    z_size = z_size//n_group
    z = torch.randn(batch_size, n_group, z_size).cuda()
    audios = torch.zeros(batch_size, mel_size*stride).cuda()

    mel = mel.unsqueeze(3)
    z = z.unsqueeze(3)

    if fp16:
        z = z.half()
        mel = mel.half()
        audios = audios.half()

    waveglow_tensors = {
        "inputs" : {'mel': mel, 'z': z},
        "outputs" : {'audio': audios}
    }

    print("Running WaveGlow with TensorRT")
    with MeasureTime(measurements, "waveglow_time"):
        run_trt_engine(waveglow_context, waveglow, waveglow_tensors)

    return audios

def infer_waveglow_onnx(waveglow_path, mel, measurements, fp16):
    import onnx
    import onnxruntime
    sess = onnxruntime.InferenceSession(waveglow_path)

    device=mel.device
    mel_size = mel.size(2)
    batch_size = mel.size(0)
    stride = 256
    n_group = 8
    z_size = mel_size*stride
    z_size = z_size//n_group
    z = torch.randn(batch_size, n_group, z_size).cuda()

    mel = mel.unsqueeze(3)
    z = z.unsqueeze(3)

    if fp16:
        z = z.half()
        mel = mel.half()

    mel = mel.cpu().numpy().copy()
    z = z.cpu().numpy().copy()

    print("Running WaveGlow with ONNX Runtime")
    with MeasureTime(measurements, "waveglow_time"):
        result = sess.run(["audio"], {
            'mel': mel,
            'z': z
        })
    audios = torch.tensor(result[0], device=device)
    return audios

def main():

    parser = argparse.ArgumentParser(
        description='TensorRT Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    # initialize CUDA state
    torch.cuda.init()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    encoder = load_engine(args.encoder, TRT_LOGGER)
    postnet = load_engine(args.postnet, TRT_LOGGER)

    if args.waveglow_ckpt != "":
        # setup denoiser using WaveGlow PyTorch checkpoint
        waveglow_ckpt = load_and_setup_model('WaveGlow', parser, args.waveglow_ckpt,
                                             True, forward_is_infer=True)
        denoiser = Denoiser(waveglow_ckpt).cuda()
        # after initialization, we don't need WaveGlow PyTorch checkpoint
        # anymore - deleting
        del waveglow_ckpt
        torch.cuda.empty_cache()

    # create TRT contexts for each engine
    encoder_context = encoder.create_execution_context()
    decoder_context = None
    if not args.decoder_onnxruntime:
        decoder_iter = load_engine(args.decoder, TRT_LOGGER)
        decoder_context = decoder_iter.create_execution_context()
    else:
        decoder_iter = args.decoder
    postnet_context = postnet.create_execution_context()

    waveglow_context = None
    if not args.waveglow_onnxruntime:
        waveglow = load_engine(args.waveglow, TRT_LOGGER)
        waveglow_context = waveglow.create_execution_context()

    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT,
                                              path.join(args.output, args.log_file)),
                            StdOutBackend(Verbosity.VERBOSE)])

    texts = []
    try:
        f = open(args.input, 'r')
        texts = f.readlines()
    except:
        print("Could not read file")
        sys.exit(1)

    measurements = {}

    sequences, sequence_lengths = prepare_input_sequence(texts)
    sequences = sequences.to(torch.int32)
    sequence_lengths = sequence_lengths.to(torch.int32)

    with MeasureTime(measurements, "latency"):
        mel, mel_lengths = infer_tacotron2_trt(encoder, decoder_iter, postnet,
                                               encoder_context, decoder_context, postnet_context,
                                               sequences, sequence_lengths, measurements, args.fp16, args.loop)
        audios = infer_waveglow_onnx(args.waveglow, mel, measurements, args.fp16) if args.waveglow_onnxruntime else \
                 infer_waveglow_trt(waveglow, waveglow_context, mel, measurements, args.fp16)

    with encoder_context, postnet_context:
        pass
        
    if decoder_context is not None:
        with decoder_context: pass
    
    if waveglow_context is not None:
        with waveglow_context: pass

    audios = audios.float()
    if args.waveglow_ckpt != "":
        with MeasureTime(measurements, "denoiser"):
            audios = denoiser(audios, strength=args.denoising_strength).squeeze(1)

    for i, audio in enumerate(audios):
        audio = audio[:mel_lengths[i]*args.stft_hop_length]
        audio = audio/torch.max(torch.abs(audio))
        audio_path = path.join(args.output, f"audio_{i}_trt.wav")
        write(audio_path, args.sampling_rate, audio.cpu().numpy())


    DLLogger.log(step=0, data={"tacotron2_encoder_latency": measurements['tacotron2_encoder_time']})
    DLLogger.log(step=0, data={"tacotron2_decoder_latency": measurements['tacotron2_decoder_time']})
    DLLogger.log(step=0, data={"tacotron2_postnet_latency": measurements['tacotron2_postnet_time']})
    DLLogger.log(step=0, data={"waveglow_latency": measurements['waveglow_time']})
    DLLogger.log(step=0, data={"latency": measurements['latency']})

    if args.waveglow_ckpt != "":
        DLLogger.log(step=0, data={"denoiser": measurements['denoiser']})
    DLLogger.flush()

    prec = "fp16" if args.fp16 else "fp32"
    latency = measurements['latency']
    throughput = audios.size(1)/latency
    log_data = f"1,{sequence_lengths[0].item()},{prec},{latency},{throughput},{mel_lengths[0].item()}\n"
    log_file = path.join(args.output, f"log_bs1_{prec}.log")
    with open(log_file, 'a') as f:
        f.write(log_data)

if __name__ == "__main__":
    main()
