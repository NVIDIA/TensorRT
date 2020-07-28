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

import types
import torch
import argparse

import sys
sys.path.append('./')

from inference import checkpoint_from_distributed, unwrap_distributed, load_and_setup_model

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--waveglow', type=str, required=True,
                        help='Full path to the WaveGlow model checkpoint file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory for the exported WaveGlow ONNX model')
    parser.add_argument('--fp16', action='store_true',
                        help='Inference with Automatic Mixed Precision')
    parser.add_argument('-s', '--sigma-infer', default=0.6, type=float,
                        help='Standard deviation of the Gaussian distribution')

    return parser


def infer_onnx(self, spect, z, sigma=0.9):

    spect = self.upsample(spect)
    # trim conv artifacts. maybe pad spec to kernel multiple
    time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
    spect = spect[:, :, :-time_cutoff]

    length_spect_group = spect.size(2)//8
    mel_dim = 80
    batch_size = spect.size(0)

    spect = spect.view((batch_size, mel_dim, length_spect_group, self.n_group))
    spect = spect.permute(0, 2, 1, 3)
    spect = spect.contiguous()
    spect = spect.view((batch_size, length_spect_group, self.n_group*mel_dim))
    spect = spect.permute(0, 2, 1)
    spect = spect.contiguous()

    audio = z[:, :self.n_remaining_channels, :]
    z = z[:, self.n_remaining_channels:self.n_group, :]
    audio = sigma*audio

    for k in reversed(range(self.n_flows)):
        n_half = int(audio.size(1) / 2)
        audio_0 = audio[:, :n_half, :]
        audio_1 = audio[:, n_half:(n_half+n_half), :]

        output = self.WN[k]((audio_0, spect))
        s = output[:, n_half:(n_half+n_half), :]
        b = output[:, :n_half, :]
        audio_1 = (audio_1 - b) / torch.exp(s)
        audio = torch.cat([audio_0, audio_1], 1)
        audio = self.convinv[k].infer(audio)

        if k % self.n_early_every == 0 and k > 0:
            audio = torch.cat((z[:, :self.n_early_size, :], audio), 1)
            z = z[:, self.n_early_size:self.n_group, :]

    audio = audio.permute(0,2,1).contiguous().view(batch_size, (length_spect_group * self.n_group))

    return audio


def export_onnx(parser, args):

    waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    fp16_run=args.fp16, cpu_run=False,
                                    forward_is_infer=False)

    # 80 mel channels, 620 mel spectrograms ~ 7 seconds of speech
    mel = torch.randn(1, 80, 620).cuda()
    stride = 256 # value from waveglow upsample
    n_group = 8
    z_size2 = (mel.size(2)*stride)//n_group
    z = torch.randn(1, n_group, z_size2).cuda()

    if args.fp16:
        mel = mel.half()
        z = z.half()
    with torch.no_grad():
        # run inference to force calculation of inverses
        waveglow.infer(mel, sigma=args.sigma_infer)

        # export to ONNX
        if args.fp16:
            waveglow = waveglow.half()

        fType = types.MethodType
        waveglow.forward = fType(infer_onnx, waveglow)

        opset_version = 11

        torch.onnx.export(waveglow, (mel, z), args.output+"/"+"waveglow.onnx",
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
