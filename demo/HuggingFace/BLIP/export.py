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
Contains logic that captures BLIP HuggingFace models into ONNX models.
Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/blip/dependencies/BLIP-export.py
"""

# TRT-HuggingFace
from NNDF.tensorrt_utils import OnnxProcessOperation, process_onnx
from NNDF.networks import Precision

from Vision2Seq.export import (
    EncoderTorchFile,
    EncoderONNXFile,
    EncoderTRTEngine,
    EncoderConverter,
    DecoderTorchFile,
    DecoderONNXFile,
    DecoderTRTEngine,
    DecoderConverter,
    Vision2SeqModelClass
)

class BLIPDecoderTorchFile(DecoderTorchFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = BLIPDecoderConverter

        super().__init__(model, network_metadata, default_converter)

class BLIPDecoderONNXFile(DecoderONNXFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = BLIPDecoderConverter

        super().__init__(model, network_metadata, default_converter)

class BLIPDecoderTRTEngine(DecoderTRTEngine):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = BLIPDecoderConverter

        super().__init__(model, network_metadata, default_converter)

class BLIPDecoderConverter(DecoderConverter):
    def __init__(self,
        torch_class=BLIPDecoderTorchFile,
        onnx_class=BLIPDecoderONNXFile,
        trt_engine_class=BLIPDecoderTRTEngine
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)

    def post_process_onnx(self, output_fpath):
        process_onnx([OnnxProcessOperation.CLAMP_WEIGHTS], output_fpath, output_fpath)

class BLIPEncoderTorchFile(EncoderTorchFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = BLIPEncoderConverter

        super().__init__(model, network_metadata, default_converter)

class BLIPEncoderONNXFile(EncoderONNXFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = BLIPEncoderConverter

        super().__init__(model, network_metadata, default_converter)

class BLIPEncoderTRTEngine(EncoderTRTEngine):

    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = BLIPEncoderConverter

        super().__init__(model, network_metadata, default_converter)

    def get_network_definition(self, network_definition):
        # BLIPEncoder tend to overflow, so we still build fp32 engine for encoder
        return network_definition

    def use_obey_precision_constraints(self):
        return False

class BLIPEncoderConverter(EncoderConverter):
    def __init__(self,
        torch_class=BLIPEncoderTorchFile,
        onnx_class=BLIPEncoderONNXFile,
        trt_engine_class=BLIPEncoderTRTEngine
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)

    def onnx_to_trt(
        self,
        output_fpath,
        input_fpath,
        network_metadata,
        profiles,
        preview_features,
        nvtx_verbose,
        timing_cache,
    ):
        # Needs to overwrite precision for onnx_to_trt
        network_metadata = network_metadata._replace(precision=Precision(fp16=False))
        return super().onnx_to_trt(
            output_fpath,
            input_fpath,
            network_metadata,
            profiles,
            preview_features,
            nvtx_verbose,
            timing_cache
        )

class BLIPModelClass(Vision2SeqModelClass):
    decoder_classes = {
        "torch": BLIPDecoderTorchFile,
        "onnx": BLIPDecoderONNXFile,
        "engine": BLIPDecoderTRTEngine,
    }

    encoder_classes = {
        "torch": BLIPEncoderTorchFile,
        "onnx": BLIPEncoderONNXFile,
        "engine": BLIPEncoderTRTEngine,
    }
