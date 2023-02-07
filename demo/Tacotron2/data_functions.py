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
from tacotron2.data_function import TextMelCollate
from tacotron2.data_function import TextMelLoader
from waveglow.data_function import MelAudioLoader
from tacotron2.data_function import batch_to_gpu as batch_to_gpu_tacotron2
from waveglow.data_function import batch_to_gpu as batch_to_gpu_waveglow


def get_collate_function(model_name, n_frames_per_step):
    if model_name == 'Tacotron2':
        collate_fn = TextMelCollate(n_frames_per_step)
    elif model_name == 'WaveGlow':
        collate_fn = torch.utils.data.dataloader.default_collate
    else:
        raise NotImplementedError(
            "unknown collate function requested: {}".format(model_name))

    return collate_fn


def get_data_loader(model_name, dataset_path, audiopaths_and_text, args):
    if model_name == 'Tacotron2':
        data_loader = TextMelLoader(dataset_path, audiopaths_and_text, args)
    elif model_name == 'WaveGlow':
        data_loader = MelAudioLoader(dataset_path, audiopaths_and_text, args)
    else:
        raise NotImplementedError(
            "unknown data loader requested: {}".format(model_name))

    return data_loader


def get_batch_to_gpu(model_name):
    if model_name == 'Tacotron2':
        batch_to_gpu = batch_to_gpu_tacotron2
    elif model_name == 'WaveGlow':
        batch_to_gpu = batch_to_gpu_waveglow
    else:
        raise NotImplementedError(
            "unknown batch_to_gpu requested: {}".format(model_name))
    return batch_to_gpu
