#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os

import numpy as np
import onnx
import torch
from diffusers import DiffusionPipeline
from onnx import numpy_helper

from demo_diffusion.model import load, optimizer
from demo_diffusion.model.lora import merge_loras


class BaseModel:
    def __init__(
        self,
        version="1.5",
        pipeline=None,
        device="cuda",
        hf_token="",
        verbose=True,
        framework_model_dir="pytorch_model",
        fp16=False,
        tf32=False,
        bf16=False,
        int8=False,
        fp8=False,
        fp4=False,
        max_batch_size=16,
        text_maxlen=77,
        embedding_dim=768,
        compression_factor=8,
    ):

        self.name = self.__class__.__name__
        self.pipeline_type = pipeline
        self.pipeline = pipeline.name
        self.version = version
        self.path = load.get_path(version, pipeline)
        self.device = device
        self.hf_token = hf_token
        self.hf_safetensor = not (pipeline.is_inpaint() and version in ("1.4", "1.5"))
        self.verbose = verbose
        self.framework_model_dir = framework_model_dir

        self.fp16 = fp16
        self.tf32 = tf32
        self.bf16 = bf16
        self.int8 = int8
        self.fp8 = fp8
        self.fp4 = fp4

        self.compression_factor = compression_factor
        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1360  # max image resolution: 1360x1360
        self.min_latent_shape = self.min_image_shape // self.compression_factor
        self.max_latent_shape = self.max_image_shape // self.compression_factor

        self.text_maxlen = text_maxlen
        self.embedding_dim = embedding_dim
        self.extra_output_names = []

        self.do_constant_folding = True

    def get_pipeline(self):
        model_opts = {"variant": "fp16", "torch_dtype": torch.float16} if self.fp16 else {}
        model_opts = {"torch_dtype": torch.bfloat16} if self.bf16 else model_opts
        return DiffusionPipeline.from_pretrained(
            self.path,
            use_safetensors=self.hf_safetensor,
            token=self.hf_token,
            **model_opts,
        ).to(self.device)

    def get_model(self, torch_inference=""):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width, static_shape):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    # Helper utility for ONNX export
    def export_onnx(
        self,
        onnx_path,
        onnx_opt_path,
        onnx_opset,
        opt_image_height,
        opt_image_width,
        custom_model=None,
        enable_lora_merge=False,
        static_shape=False,
        lora_loader=None,
        dynamo=False,
    ):
        onnx_opt_graph = None
        # Export optimized ONNX model (if missing)
        if not os.path.exists(onnx_opt_path):
            if not os.path.exists(onnx_path):
                print(f"[I] Exporting ONNX model: {onnx_path}")

                def export_onnx(model):
                    if enable_lora_merge:
                        assert lora_loader is not None
                        model = merge_loras(model, lora_loader)

                    export_kwargs = {}
                    if dynamo:
                        export_kwargs["dynamic_shapes"] = self.get_dynamic_axes()
                    else:
                        export_kwargs["dynamic_axes"] = self.get_dynamic_axes()
                    inputs = self.get_sample_input(1, opt_image_height, opt_image_width, static_shape)
                    torch.onnx.export(
                        model,
                        inputs,
                        onnx_path,
                        export_params=True,
                        opset_version=onnx_opset,
                        do_constant_folding=self.do_constant_folding,
                        input_names=self.get_input_names(),
                        output_names=self.get_output_names(),
                        verbose=False,
                        dynamo=dynamo,
                        **export_kwargs,
                    )

                if custom_model:
                    with torch.inference_mode():
                        export_onnx(custom_model)
                else:
                    # WAR: Enable autocast for BF16 Stable Cascade pipeline
                    do_autocast = True if self.version == "cascade" and self.bf16 else False
                    with torch.inference_mode(), torch.autocast("cuda", enabled=do_autocast):
                        export_onnx(self.get_model())
            else:
                print(f"[I] Found cached ONNX model: {onnx_path}")

            print(f"[I] Optimizing ONNX model: {onnx_opt_path}")
            onnx_opt_graph = self.optimize(onnx.load(onnx_path))
            if load.onnx_graph_needs_external_data(onnx_opt_graph):
                onnx.save_model(
                    onnx_opt_graph,
                    onnx_opt_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False,
                )
            else:
                onnx.save(onnx_opt_graph, onnx_opt_path)
        else:
            print(f"[I] Found cached optimized ONNX model: {onnx_opt_path} ")

    # Helper utility for weights map
    def export_weights_map(self, onnx_opt_path, weights_map_path):
        if not os.path.exists(weights_map_path):
            onnx_opt_dir = os.path.dirname(onnx_opt_path)
            onnx_opt_model = onnx.load(onnx_opt_path)
            state_dict = self.get_model().state_dict()
            # Create initializer data hashes
            initializer_hash_mapping = {}
            for initializer in onnx_opt_model.graph.initializer:
                initializer_data = numpy_helper.to_array(initializer, base_dir=onnx_opt_dir).astype(np.float16)
                initializer_hash = hash(initializer_data.data.tobytes())
                initializer_hash_mapping[initializer.name] = (initializer_hash, initializer_data.shape)

            weights_name_mapping = {}
            weights_shape_mapping = {}
            # set to keep track of initializers already added to the name_mapping dict
            initializers_mapped = set()
            for wt_name, wt in state_dict.items():
                # get weight hash
                wt = wt.cpu().detach().numpy().astype(np.float16)
                wt_hash = hash(wt.data.tobytes())
                wt_t_hash = hash(np.transpose(wt).data.tobytes())

                for initializer_name, (initializer_hash, initializer_shape) in initializer_hash_mapping.items():
                    # Due to constant folding, some weights are transposed during export
                    # To account for the transpose op, we compare the initializer hash to the
                    # hash for the weight and its transpose
                    if wt_hash == initializer_hash or wt_t_hash == initializer_hash:
                        # The assert below ensures there is a 1:1 mapping between
                        # PyTorch and ONNX weight names. It can be removed in cases where 1:many
                        # mapping is found and name_mapping[wt_name] = list()
                        assert initializer_name not in initializers_mapped
                        weights_name_mapping[wt_name] = initializer_name
                        initializers_mapped.add(initializer_name)
                        is_transpose = False if wt_hash == initializer_hash else True
                        weights_shape_mapping[wt_name] = (initializer_shape, is_transpose)

                # Sanity check: Were any weights not matched
                if wt_name not in weights_name_mapping:
                    print(f"[I] PyTorch weight {wt_name} not matched with any ONNX initializer")
            print(f"[I] {len(weights_name_mapping.keys())} PyTorch weights were matched with ONNX initializers")
            assert weights_name_mapping.keys() == weights_shape_mapping.keys()
            with open(weights_map_path, "w") as fp:
                json.dump([weights_name_mapping, weights_shape_mapping], fp)
        else:
            print(f"[I] Found cached weights map: {weights_map_path} ")

    def optimize(self, onnx_graph, return_onnx=True, **kwargs):
        opt = optimizer.Optimizer(onnx_graph, verbose=self.verbose, version=self.version)
        opt.info(self.name + ": original")
        opt.cleanup()
        opt.info(self.name + ": cleanup")
        if kwargs.get("modify_fp8_graph", False):
            is_fp16_io = kwargs.get("is_fp16_io", True)
            opt.modify_fp8_graph(is_fp16_io=is_fp16_io)
            opt.info(self.name + ": modify fp8 graph")
        if self.version.startswith("flux.1") and self.fp8:
            opt.flux_convert_rope_weight_type()
            opt.info(self.name + ": convert rope weight type for fp8 flux")
        opt.fold_constants()
        opt.info(self.name + ": fold constants")
        opt.infer_shapes()
        opt.info(self.name + ": shape inference")
        if kwargs.get("fuse_mha_qkv_int8", False):
            opt.fuse_mha_qkv_int8_sq()
            opt.info(self.name + ": fuse QKV nodes")
        onnx_opt_graph = opt.cleanup(return_onnx=return_onnx)
        opt.info(self.name + ": finished")
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        latent_height = image_height // self.compression_factor
        latent_width = image_width // self.compression_factor
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // self.compression_factor
        latent_width = image_width // self.compression_factor
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )
