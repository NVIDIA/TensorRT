#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Contains logic that captures HuggingFace models into ONNX models.
"""

from Seq2Seq.export import (
    DecoderTorchFile,
    DecoderONNXFile,
    DecoderTRTEngine,
    Seq2SeqModelClass
)

class OPTModelClass(Seq2SeqModelClass):
    """
    A class to track which class to use for each model type.
    """

    decoder_classes = {
        "torch": DecoderTorchFile,
        "onnx": DecoderONNXFile,
        "engine": DecoderTRTEngine
    }
