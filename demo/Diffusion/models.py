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

from collections import OrderedDict
from copy import deepcopy
from cuda import cudart
from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
import numpy as np
import onnx
from onnx import shape_inference
import onnx_graphsurgeon as gs
import os
from polygraphy.backend.onnx.loader import fold_constants
import shutil
import tempfile
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

class Optimizer():
    def __init__(
        self,
        onnx_graph,
        verbose=False
    ):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            temp_dir = tempfile.TemporaryDirectory().name
            os.makedirs(temp_dir, exist_ok=True)
            onnx_orig_path = os.path.join(temp_dir, 'model.onnx')
            onnx_inferred_path = os.path.join(temp_dir, 'inferred.onnx')
            onnx.save_model(onnx_graph,
                onnx_orig_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False)
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path)
            onnx_graph = onnx.load(onnx_inferred_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def clip_add_hidden_states(self, return_onnx=False):
        hidden_layers = -1
        onnx_graph = gs.export_onnx(self.graph)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                name = onnx_graph.graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(int(name.split(".")[1].split("/")[0]), hidden_layers)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                if onnx_graph.graph.node[i].output[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers-1):
                    onnx_graph.graph.node[i].output[j] = "hidden_states"
            for j in range(len(onnx_graph.graph.node[i].input)):
                if onnx_graph.graph.node[i].input[j] == "/text_model/encoder/layers.{}/Add_1_output_0".format(hidden_layers-1):
                    onnx_graph.graph.node[i].input[j] = "hidden_states"
        if return_onnx:
            return onnx_graph

def get_controlnets_path(controlnet_list):
    '''
    Currently ControlNet 1.0 is supported.
    '''
    if controlnet_list is None:
        return None
    return ["lllyasviel/sd-controlnet-" + controlnet for controlnet in controlnet_list]

def get_path(version, pipeline, controlnet=None):

    if controlnet is not None:
        return ["lllyasviel/sd-controlnet-" + modality for modality in controlnet]
    
    if version == "1.4":
        if pipeline.is_inpaint():
            return "runwayml/stable-diffusion-inpainting"
        else:
            return "CompVis/stable-diffusion-v1-4"
    elif version == "1.5":
        if pipeline.is_inpaint():
            return "runwayml/stable-diffusion-inpainting"
        else:
            return "runwayml/stable-diffusion-v1-5"
    elif version == "2.0-base":
        if pipeline.is_inpaint():
            return "stabilityai/stable-diffusion-2-inpainting"
        else:
            return "stabilityai/stable-diffusion-2-base"
    elif version == "2.0":
        if pipeline.is_inpaint():
            return "stabilityai/stable-diffusion-2-inpainting"
        else:
            return "stabilityai/stable-diffusion-2"
    elif version == "2.1":
        return "stabilityai/stable-diffusion-2-1"
    elif version == "2.1-base":
        return "stabilityai/stable-diffusion-2-1-base"
    elif version == 'xl-1.0':
        if pipeline.is_sd_xl_base():
            return "stabilityai/stable-diffusion-xl-base-1.0"
        elif pipeline.is_sd_xl_refiner():
            return "stabilityai/stable-diffusion-xl-refiner-1.0"
        else:
            raise ValueError(f"Unsupported SDXL 1.0 pipeline {pipeline.name}")
    else:
        raise ValueError(f"Incorrect version {version}")

def get_clip_embedding_dim(version, pipeline):
    if version in ("1.4", "1.5"):
        return 768
    elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
        return 1024
    elif version in ("xl-1.0") and pipeline.is_sd_xl_base():
        return 768
    else:
        raise ValueError(f"Invalid version {version} + pipeline {pipeline}")

def get_clipwithproj_embedding_dim(version, pipeline):
    if version in ("xl-1.0"):
        return 1280
    else:
        raise ValueError(f"Invalid version {version} + pipeline {pipeline}")

def get_unet_embedding_dim(version, pipeline):
    if version in ("1.4", "1.5"):
        return 768
    elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
        return 1024
    elif version in ("xl-1.0") and pipeline.is_sd_xl_base():
        return 2048
    elif version in ("xl-1.0") and pipeline.is_sd_xl_refiner():
        return 1280
    else:
        raise ValueError(f"Invalid version {version} + pipeline {pipeline}")

class BaseModel():
    def __init__(self,
        version='1.5',
        pipeline=None,
        hf_token='',
        device='cuda',
        verbose=True,
        fp16=False,
        max_batch_size=16,
        text_maxlen=77,
        embedding_dim=768,
        controlnet=None
    ):

        self.name = self.__class__.__name__
        self.pipeline = pipeline.name
        self.version = version
        self.hf_token = hf_token
        self.hf_safetensor = pipeline.is_sd_xl()
        self.device = device
        self.verbose = verbose
        self.path = get_path(version, pipeline, controlnet)

        self.fp16 = fp16

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256   # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.text_maxlen = text_maxlen
        self.embedding_dim = embedding_dim
        self.extra_output_names = []

    def get_model(self, framework_model_dir):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.cleanup()
        opt.info(self.name + ': cleanup')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ': finished')
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width)

class CLIP(BaseModel):
    def __init__(self,
        version,
        pipeline,
        hf_token,
        device,
        verbose,
        max_batch_size,
        embedding_dim,
        output_hidden_states=False,
        subfolder="text_encoder"
    ):
        super(CLIP, self).__init__(version, pipeline, hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size, embedding_dim=embedding_dim)
        self.subfolder = subfolder

        # Output the final hidden state
        if output_hidden_states:
            self.extra_output_names = ['hidden_states']

    def get_model(self, framework_model_dir):
        clip_model_dir = os.path.join(framework_model_dir, self.version, self.pipeline, "text_encoder")
        if not os.path.exists(clip_model_dir):
            model = CLIPTextModel.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                use_auth_token=self.hf_token).to(self.device)
            model.save_pretrained(clip_model_dir)
        else:
            print(f"[I] Load CLIP pytorch model from: {clip_model_dir}")
            model = CLIPTextModel.from_pretrained(clip_model_dir).to(self.device)
        return model

    def get_input_names(self):
        return ['input_ids']

    def get_output_names(self):
       return ['text_embeddings']

    def get_dynamic_axes(self):
        return {
            'input_ids': {0: 'B'},
            'text_embeddings': {0: 'B'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim)
        }
        if 'hidden_states' in self.extra_output_names:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)
        return output

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.select_outputs([0]) # delete graph output#1
        opt.cleanup()
        opt.info(self.name + ': remove output[1]')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        opt.select_outputs([0], names=['text_embeddings']) # rename network output
        opt.info(self.name + ': remove output[0]')
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        if 'hidden_states' in self.extra_output_names:
            opt_onnx_graph = opt.clip_add_hidden_states(return_onnx=True)
            opt.info(self.name + ': added hidden_states')
        opt.info(self.name + ': finished')
        return opt_onnx_graph

def make_CLIP(version, pipeline, hf_token, device, verbose, max_batch_size, output_hidden_states=False, subfolder="text_encoder"):
    return CLIP(version, pipeline, hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size, embedding_dim=get_clip_embedding_dim(version, pipeline), output_hidden_states=output_hidden_states, subfolder=subfolder)


class CLIPWithProj(CLIP):
    def __init__(self,
        version,
        pipeline,
        hf_token,
        device='cuda',
        verbose=True,
        max_batch_size=16,
        output_hidden_states=False,
        subfolder="text_encoder_2"):

        super(CLIPWithProj, self).__init__(version, pipeline, hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size, embedding_dim=get_clipwithproj_embedding_dim(version, pipeline), output_hidden_states=output_hidden_states)
        self.subfolder = subfolder

    def get_model(self, framework_model_dir):
        clip_model_dir = os.path.join(framework_model_dir, self.version, self.pipeline, "text_encoder_2")
        if not os.path.exists(clip_model_dir):
            model = CLIPTextModelWithProjection.from_pretrained(self.path,
                subfolder=self.subfolder,
                use_safetensors=self.hf_safetensor,
                use_auth_token=self.hf_token).to(self.device)
            model.save_pretrained(clip_model_dir)
        else:
            print(f"[I] Load CLIP pytorch model from: {clip_model_dir}")
            model = CLIPTextModelWithProjection.from_pretrained(clip_model_dir).to(self.device)
        return model

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        output = {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.embedding_dim)
        }
        if 'hidden_states' in self.extra_output_names:
            output["hidden_states"] = (batch_size, self.text_maxlen, self.embedding_dim)

        return output

def make_CLIPWithProj(version, pipeline, hf_token, device, verbose, max_batch_size, subfolder="text_encoder_2", output_hidden_states=False):
    return CLIPWithProj(version, pipeline, hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size, subfolder=subfolder, output_hidden_states=output_hidden_states)

class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self, unet, controlnets) -> None:
        super().__init__()
        self.unet = unet
        self.controlnets = controlnets
        
    def forward(self, sample, timestep, encoder_hidden_states, images, controlnet_scales):
        for i, (image, conditioning_scale, controlnet) in enumerate(zip(images, controlnet_scales, self.controlnets)):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=image,
                return_dict=False,
            )

            down_samples = [
                    down_sample * conditioning_scale
                    for down_sample in down_samples
                ]
            mid_sample *= conditioning_scale
            
            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample
        
        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample
        )
        return noise_pred

class UNet(BaseModel):
    def __init__(self,
        version,
        pipeline,
        hf_token,
        device='cuda',
        verbose=True,
        fp16=False,
        max_batch_size=16,
        text_maxlen=77,
        unet_dim=4,
        controlnet=None
    ):

        super(UNet, self).__init__(version, pipeline, hf_token, fp16=fp16, device=device, verbose=verbose, max_batch_size=max_batch_size, text_maxlen=text_maxlen, embedding_dim=get_unet_embedding_dim(version, pipeline))
        self.unet_dim = unet_dim
        self.controlnet = controlnet

    def get_model(self, framework_model_dir):
        model_opts = {'variant': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        if self.controlnet:
            unet_model = UNet2DConditionModel.from_pretrained(self.path,
                subfolder="unet",
                use_safetensors=self.hf_safetensor,
                use_auth_token=self.hf_token,
                **model_opts).to(self.device)

            cnet_model_opts = {'torch_dtype': torch.float16} if self.fp16 else {}
            controlnets = torch.nn.ModuleList([ControlNetModel.from_pretrained(path, **cnet_model_opts).to(self.device) for path in self.controlnet])
            # FIXME - cache UNet2DConditionControlNetModel
            model = UNet2DConditionControlNetModel(unet_model, controlnets)
        else:
            unet_model_dir = os.path.join(framework_model_dir, self.version, self.pipeline, "unet")
            if not os.path.exists(unet_model_dir):
                model = UNet2DConditionModel.from_pretrained(self.path,
                    subfolder="unet",
                    use_safetensors=self.hf_safetensor,
                    use_auth_token=self.hf_token,
                    **model_opts).to(self.device)
                model.save_pretrained(unet_model_dir)
            else:
                print(f"[I] Load UNet pytorch model from: {unet_model_dir}")
                model = UNet2DConditionModel.from_pretrained(unet_model_dir).to(self.device)
        return model

    def get_input_names(self):
        if self.controlnet is None:
            return ['sample', 'timestep', 'encoder_hidden_states']
        else:    
            return ['sample', 'timestep', 'encoder_hidden_states', 'images', 'controlnet_scales']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        if self.controlnet is None:
            return {
                'sample': {0: '2B', 2: 'H', 3: 'W'},
                'encoder_hidden_states': {0: '2B'},
                'latent': {0: '2B', 2: 'H', 3: 'W'}
            }
        else:
            return {
                'sample': {0: '2B', 2: 'H', 3: 'W'},
                'encoder_hidden_states': {0: '2B'},
                'images': {1: '2B', 3: '8H', 4: '8W'},
                'latent': {0: '2B', 2: 'H', 3: 'W'}
            }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        if self.controlnet is None:
            return {
                'sample': [(2*min_batch, self.unet_dim, min_latent_height, min_latent_width), (2*batch_size, self.unet_dim, latent_height, latent_width), (2*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
                'encoder_hidden_states': [(2*min_batch, self.text_maxlen, self.embedding_dim), (2*batch_size, self.text_maxlen, self.embedding_dim), (2*max_batch, self.text_maxlen, self.embedding_dim)]
            }
        else:
            return {
                'sample': [(2*min_batch, self.unet_dim, min_latent_height, min_latent_width), 
                           (2*batch_size, self.unet_dim, latent_height, latent_width), 
                           (2*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
                'encoder_hidden_states': [(2*min_batch, self.text_maxlen, self.embedding_dim), 
                                          (2*batch_size, self.text_maxlen, self.embedding_dim), 
                                          (2*max_batch, self.text_maxlen, self.embedding_dim)],
                'images': [(len(self.controlnet), 2*min_batch, 3, min_image_height, min_image_width), 
                          (len(self.controlnet), 2*batch_size, 3, image_height, image_width), 
                          (len(self.controlnet), 2*max_batch, 3, max_image_height, max_image_width)]
            }


    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        if self.controlnet is None:
            return {
                'sample': (2*batch_size, self.unet_dim, latent_height, latent_width),
                'encoder_hidden_states': (2*batch_size, self.text_maxlen, self.embedding_dim),
                'latent': (2*batch_size, 4, latent_height, latent_width)
            }
        else:
            return {
                'sample': (2*batch_size, self.unet_dim, latent_height, latent_width),
                'encoder_hidden_states': (2*batch_size, self.text_maxlen, self.embedding_dim),
                'images': (len(self.controlnet), 2*batch_size, 3, image_height, image_width), 
                'latent': (2*batch_size, 4, latent_height, latent_width)
                }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        if self.controlnet is None:
            return (
                torch.randn(batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
                torch.tensor([1.], dtype=torch.float32, device=self.device),
                torch.randn(batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device)
            )
        else:
            return (
                torch.randn(batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
                torch.tensor(999, dtype=torch.float32, device=self.device),
                torch.randn(batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
                torch.randn(len(self.controlnet), batch_size, 3, image_height, image_width, dtype=dtype, device=self.device),
                torch.randn(len(self.controlnet), dtype=dtype, device=self.device)
            )

def make_UNet(version, pipeline, hf_token, device, verbose, max_batch_size, controlnet=None):
    # Disable torch SDPA
    if hasattr(F, "scaled_dot_product_attention"):
        delattr(F, "scaled_dot_product_attention")

    return UNet(version, pipeline, hf_token, fp16=True, device=device, verbose=verbose,
            max_batch_size=max_batch_size, unet_dim=(9 if pipeline.is_inpaint() else 4),
            controlnet=get_controlnets_path(controlnet))

class UNetXL(BaseModel):
    def __init__(self,
        version,
        pipeline,
        hf_token,
        fp16=False,
        device='cuda',
        verbose=True,
        max_batch_size=16,
        text_maxlen=77,
        unet_dim=4,
        time_dim=6
    ):
        super(UNetXL, self).__init__(version, pipeline, hf_token, fp16=fp16, device=device, verbose=verbose, max_batch_size=max_batch_size, text_maxlen=text_maxlen, embedding_dim=get_unet_embedding_dim(version, pipeline))
        self.unet_dim = unet_dim
        self.time_dim = time_dim

    def get_model(self, framework_model_dir):
        model_opts = {'variant': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        unet_model_dir = os.path.join(framework_model_dir, self.version, self.pipeline, "unet")
        if not os.path.exists(unet_model_dir):
            model = UNet2DConditionModel.from_pretrained(self.path,
                subfolder="unet",
                use_safetensors=self.hf_safetensor,
                use_auth_token=self.hf_token,
                **model_opts).to(self.device)
            model.save_pretrained(unet_model_dir)
        else:
            print(f"[I] Load UNet pytorch model from: {unet_model_dir}")
            model = UNet2DConditionModel.from_pretrained(unet_model_dir).to(self.device)
        return model

    def get_input_names(self):
        return ['sample', 'timestep', 'encoder_hidden_states', 'text_embeds', 'time_ids']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            'sample': {0: '2B', 2: 'H', 3: 'W'},
            'encoder_hidden_states': {0: '2B'},
            'latent': {0: '2B', 2: 'H', 3: 'W'},
            'text_embeds': {0: '2B'},
            'time_ids': {0: '2B'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'sample': [(2*min_batch, self.unet_dim, min_latent_height, min_latent_width), (2*batch_size, self.unet_dim, latent_height, latent_width), (2*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
            'encoder_hidden_states': [(2*min_batch, self.text_maxlen, self.embedding_dim), (2*batch_size, self.text_maxlen, self.embedding_dim), (2*max_batch, self.text_maxlen, self.embedding_dim)],
            'text_embeds': [(2*min_batch, 1280), (2*batch_size, 1280), (2*max_batch, 1280)],
            'time_ids': [(2*min_batch, self.time_dim), (2*batch_size, self.time_dim), (2*max_batch, self.time_dim)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (2*batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (2*batch_size, self.text_maxlen, self.embedding_dim),
            'latent': (2*batch_size, 4, latent_height, latent_width),
            'text_embeds': (2*batch_size, 1280),
            'time_ids': (2*batch_size, self.time_dim)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(2*batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
            torch.tensor([1.], dtype=torch.float32, device=self.device),
            torch.randn(2*batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            {
                'added_cond_kwargs': {
                    'text_embeds': torch.randn(2*batch_size, 1280, dtype=dtype, device=self.device),
                    'time_ids' : torch.randn(2*batch_size, self.time_dim, dtype=dtype, device=self.device)
                }
            }
        )

def make_UNetXL(version, pipeline, hf_token, device, verbose, max_batch_size):
    # Disable torch SDPA
    if hasattr(F, "scaled_dot_product_attention"):
        delattr(F, "scaled_dot_product_attention")
    return UNetXL(version, pipeline, hf_token, fp16=True,  device=device, verbose=verbose,
                max_batch_size=max_batch_size, unet_dim=4, time_dim=(5 if pipeline.is_sd_xl_refiner() else 6))

class VAE(BaseModel):
    def __init__(self,
        version,
        pipeline,
        hf_token,
        device,
        verbose,
        max_batch_size,
    ):
        super(VAE, self).__init__(version, pipeline, hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size)

    def get_model(self, framework_model_dir):
        vae_decoder_model_path = os.path.join(framework_model_dir, self.version, self.pipeline, "vae_decoder")
        if not os.path.exists(vae_decoder_model_path):
            vae = AutoencoderKL.from_pretrained(self.path,
                subfolder="vae",
                use_safetensors=self.hf_safetensor,
                use_auth_token=self.hf_token).to(self.device)
            vae.save_pretrained(vae_decoder_model_path)
        else:
            print(f"[I] Load VAE decoder pytorch model from: {vae_decoder_model_path}")
            vae = AutoencoderKL.from_pretrained(vae_decoder_model_path).to(self.device)
        vae.forward = vae.decode
        return vae

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
       return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [(min_batch, 4, min_latent_height, min_latent_width), (batch_size, 4, latent_height, latent_width), (max_batch, 4, max_latent_height, max_latent_width)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'latent': (batch_size, 4, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)


def make_VAE(version, pipeline, hf_token, device, verbose, max_batch_size):
    return VAE(version, pipeline, hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size)

class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, version, pipeline, hf_token, device, path, framework_model_dir, hf_safetensor=False):
        super().__init__()
        vae_encoder_model_dir = os.path.join(framework_model_dir, version, pipeline, "vae_encoder")
        if not os.path.exists(vae_encoder_model_dir):
            self.vae_encoder = AutoencoderKL.from_pretrained(path,
                subfolder="vae",
                use_safetensors=hf_safetensor,
                use_auth_token=hf_token).to(device)
            self.vae_encoder.save_pretrained(vae_encoder_model_dir)
        else:
            print(f"[I] Load VAE encoder pytorch model from: {vae_encoder_model_dir}")
            self.vae_encoder = AutoencoderKL.from_pretrained(vae_encoder_model_dir).to(device)

    def forward(self, x):
        return self.vae_encoder.encode(x).latent_dist.sample()


class VAEEncoder(BaseModel):
    def __init__(self,
        version,
        pipeline,
        hf_token,
        device,
        verbose,
        max_batch_size,
    ):
        super(VAEEncoder, self).__init__(version, pipeline, hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size)

    def get_model(self, framework_model_dir):
        vae_encoder = TorchVAEEncoder(self.version, self.pipeline, self.hf_token, self.device, self.path, framework_model_dir, hf_safetensor=self.hf_safetensor)
        return vae_encoder

    def get_input_names(self):
        return ['images']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            'images': {0: 'B', 2: '8H', 3: '8W'},
            'latent': {0: 'B', 2: 'H', 3: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, _, _, _, _ = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            'images': [(min_batch, 3, min_image_height, min_image_width), (batch_size, 3, image_height, image_width), (max_batch, 3, max_image_height, max_image_width)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'images': (batch_size, 3, image_height, image_width),
            'latent': (batch_size, 4, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device)

def make_VAEEncoder(version, pipeline, hf_token, device, verbose, max_batch_size):
    return VAEEncoder(version, pipeline, hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size)

def make_tokenizer(version, pipeline, hf_token, framework_model_dir, subfolder="tokenizer"):
    tokenizer_model_dir = os.path.join(framework_model_dir, version, pipeline.name, subfolder)
    if not os.path.exists(tokenizer_model_dir):
        model = CLIPTokenizer.from_pretrained(get_path(version, pipeline),
                subfolder=subfolder,
                use_safetensors=pipeline.is_sd_xl(),
                use_auth_token=hf_token)
        model.save_pretrained(tokenizer_model_dir)
    else:
        print(f"[I] Load tokenizer pytorch model from: {tokenizer_model_dir}")
        model = CLIPTokenizer.from_pretrained(tokenizer_model_dir)
    return model
