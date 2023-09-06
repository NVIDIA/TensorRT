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
Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py
"""

# tensorrt
import tensorrt as trt

# torch
import torch
from torch.nn import Module

# Huggingface
from transformers import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from NNDF.networks import NetworkMetadata
from NNDF.logger import G_LOGGER
from NNDF.models import (
    TRTEngineFile,
    TorchModelFile,
    ONNXModelFile,
    ModelFileConverter,
)

OPSET = 17
TRAINING_MODE = torch.onnx.TrainingMode.EVAL
CONSTANT_FOLDING = True

# Encoder File Encoding #
class EncoderTorchFile(TorchModelFile):

    class TorchModule(Module):
        def __init__(self, model: Module):
            super().__init__()
            self.encoder = model.vision_model
            self.device = model.device

        def forward(self, *input, **kwargs):
            image_embeds = self.encoder(*input, **kwargs)[0]
            return BaseModelOutput(last_hidden_state = image_embeds)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = EncoderConverter

        super().__init__(model, default_converter, network_metadata)

class EncoderONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = EncoderConverter

        super().__init__(model, default_converter, network_metadata)

class EncoderTRTEngine(TRTEngineFile):

    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = EncoderConverter

        super().__init__(model, default_converter, network_metadata)

    def get_network_definition(self, network_definition):

        if self.network_metadata.precision.fp16:
            for i in range(network_definition[1].num_inputs):
                t = network_definition[1].get_input(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

            for i in range(network_definition[1].num_outputs):
                t = network_definition[1].get_output(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

        return network_definition

    def use_obey_precision_constraints(self):
        return self.network_metadata.precision.fp16

class EncoderConverter(ModelFileConverter):
    def __init__(self,
        torch_class=EncoderTorchFile,
        onnx_class=EncoderONNXFile,
        trt_engine_class=EncoderTRTEngine
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)


    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata, config
    ):
        """
        Exports a given huggingface Vision2Seq to encoder architecture only.
        Inspired by https://github.com/onnx/models/blob/main/text/machine_comprehension/t5/dependencies/T5-export.py

        Args:
            output_fpath (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            Tuple[str]: Names of generated models
        """
        device = model.device
        inputs = config.get_input_dims()[config.NETWORK_ENCODER_SEGMENT_NAME]
        outputs = config.get_output_dims()[config.NETWORK_ENCODER_SEGMENT_NAME]

        pixel_values = torch.ones([1, 3, config.image_size, config.image_size]).to(device).to(model.dtype) # batch_size = 1 for dummpy input
        # Exporting the decoder requires a basic instance of the encoder

        # Create one temporarily
        simplified_encoder = config.encoder_classes["torch"].TorchModule(model)

        # Exports to ONNX
        torch.onnx.export(
            simplified_encoder,
            pixel_values,
            output_fpath,
            do_constant_folding=CONSTANT_FOLDING,
            opset_version=OPSET,
            input_names=inputs.get_names(),
            output_names=outputs.get_names(),
            dynamic_axes={
                **inputs.get_torch_dynamic_axis_encoding(),
                **outputs.get_torch_dynamic_axis_encoding(),
            },
            training=torch.onnx.TrainingMode.EVAL,
        )

        if network_metadata.precision.fp16:
            self.post_process_onnx(output_fpath)

        return self.onnx_class(model = output_fpath, network_metadata = network_metadata)

# Decoder File Encoding #
class DecoderTorchFile(TorchModelFile):
    class TorchModule(Module, GenerationMixin):
        """
        A simplied definition of Vision2Seq Decoder without support for loss.
        Decoder with lm-head attached.
        """

        def __init__(self, model: Module):
            super().__init__()
            self.decoder = model.text_decoder.bert
            self.lm_head = model.text_decoder.cls

            self.config = model.text_decoder.config

            # Required for decoder generation for HuggingFace
            if hasattr(model.text_decoder, "main_input_name"):
                self.main_input_name = model.main_input_name
            if hasattr(model.text_decoder, "generation_config"):
                self.generation_config = model.text_decoder.generation_config

            self.device = model.device

        def can_generate(self):
            return True

        def prepare_inputs_for_generation(
            self,
            input_ids,
            attention_mask = None,
            past_key_values = None,
            use_cache = None,
            image_embeds = None,
            encoder_outputs = None, # For consistency to Seq2Seq input args (useless here)
        ):
            
            attention_mask = None # Do not let generation utility to generate attention_masks

            if past_key_values is not None:
                input_ids = input_ids[:, -1:]

            input_dict = {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "image_embeds": image_embeds,
                "use_cache": use_cache,
            }

            if not self.config.is_encoder_decoder:
                input_dict["attention_mask"] = attention_mask

            return input_dict

        def forward(
            self,
            input_ids,
            attention_mask = None,
            image_embeds = None,
            past_key_values = None,
            use_cache = None,
            **kwargs,
        ):
            extra_args = {}
            extra_args["is_decoder"] = self.config.is_decoder # Necessary for BERT-style decoder for BLIP

            decoder_outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=image_embeds,
                use_cache=use_cache,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                **extra_args,
                **kwargs
            )

            sequence_output = decoder_outputs[0]
            logits = self.lm_head(sequence_output) if self.lm_head is not None else sequence_output
            past_key_values = decoder_outputs[1] if use_cache else None

            return Seq2SeqLMOutput(
                logits=logits,
                past_key_values=past_key_values
            )

        def _reorder_cache(self, past_key_values, beam_idx):
            # if decoder past is not included in output
            # speedy decoding is disabled and no need to reorder
            if past_key_values is None:
                G_LOGGER.warning("You might want to consider setting `use_cache=True` to speed up decoding")
                return past_key_values

            reordered_decoder_past = ()
            for layer_past_states in past_key_values:
                # get the correct batch idx from layer past batch dim
                # batch dim of `past` is at 2nd position
                reordered_layer_past_states = ()
                for layer_past_state in layer_past_states:
                    # need to set correct `past` for each of the four key / value states
                    reordered_layer_past_states = reordered_layer_past_states + (
                        layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                    )

                assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
                assert len(reordered_layer_past_states) == len(layer_past_states)

                reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
            return reordered_decoder_past

    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = DecoderConverter
        super().__init__(model, default_converter, network_metadata)

class DecoderONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = DecoderConverter
        super().__init__(model, default_converter, network_metadata)

class DecoderTRTEngine(TRTEngineFile):
    def __init__(self, model, network_metadata = None, default_converter = None):
        if default_converter is None:
            default_converter = DecoderConverter
        super().__init__(model, default_converter, network_metadata)

    def get_network_definition(self, network_definition):

        if self.network_metadata.precision.fp16:
            for i in range(network_definition[1].num_inputs):
                t = network_definition[1].get_input(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

            for i in range(network_definition[1].num_outputs):
                t = network_definition[1].get_output(i)
                if t.dtype == trt.float32:
                    t.dtype = trt.float16

        return self.set_layer_precisions(network_definition)

    def set_layer_precisions(self, network_definition):
        """
        Users could customize this function for different model if mixed precision is needed.
        """
        return network_definition

    def use_obey_precision_constraints(self):
        return self.network_metadata.precision.fp16

class DecoderConverter(ModelFileConverter):
    def __init__(self,
        torch_class=DecoderTorchFile,
        onnx_class=DecoderONNXFile,
        trt_engine_class=DecoderTRTEngine,
    ):
        super().__init__(torch_class=torch_class, onnx_class=onnx_class, trt_engine_class=trt_engine_class)

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata, config
    ):
        """
        Exports a given huggingface Vision2Seq to decoder architecture only.
        Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/Vision2Seq-export.py

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            DecoderONNXFile: ONNX decoder object.
        """
        # Adding a device parameter to the class may help
        device = model.device
        pixel_values = torch.ones([10, 3, config.image_size, config.image_size]).to(device).to(model.dtype) # batch_size = 1 for dummpy input
        # Exporting the decoder requires a basic instance of the encoder
        # Create one temporarily
        simplified_encoder = config.encoder_classes["torch"].TorchModule(model)
        vision_outputs = simplified_encoder(pixel_values = pixel_values)
        image_embeds = vision_outputs[0]

        input_ids = (
            torch.LongTensor([[config.text_config.bos_token_id, config.text_config.eos_token_id]])
            .repeat(10, 1) # batch_size = 10 for dummpy input
            .to(device)
        )
        input_ids[:, 0] = config.text_config.bos_token_id
        
        if config.use_mask:
            attention_mask = torch.ones_like(input_ids).to(device)
        else:
            attention_mask = None

        # Exports to ONNX
        decoder_with_lm_head = self.torch_class.TorchModule(model)

        # This would require the use of config
        inputs = config.get_input_dims()[config.NETWORK_DECODER_SEGMENT_NAME]
        outputs = config.get_output_dims()[config.NETWORK_DECODER_SEGMENT_NAME]

        # Exports to ONNX
        if not network_metadata.use_cache:
            # This code allows for huggingface compatible torch class to use onnx exporter
            old_forward = decoder_with_lm_head.forward
            def _export_forward(input_ids, attention_mask, image_embeds):
                result = old_forward(input_ids, attention_mask=attention_mask, image_embeds=image_embeds, use_cache=False)
                return result[0]

            decoder_with_lm_head.forward = _export_forward
            torch.onnx.export(
                decoder_with_lm_head,
                (input_ids, attention_mask, image_embeds),
                output_fpath,
                do_constant_folding=CONSTANT_FOLDING,
                opset_version=OPSET,
                input_names=inputs.get_names(),
                output_names=outputs.get_names(),
                dynamic_axes={
                    **inputs.get_torch_dynamic_axis_encoding(),
                    **outputs.get_torch_dynamic_axis_encoding(),
                },
                training=TRAINING_MODE,
            )
        else:
            # decoder_attention_mask = None
            # decoder_input_ids = input_ids[:,-1:]
            # if config.use_mask:
            #     decoder_attention_mask = torch.tensor([[1]]).to(device)
            decoder_attention_mask = None
            if config.is_encoder_decoder:
                # We need to use input_ids[:,-1] for encoder/decoder models for kv cache,
                # BART/T5 only work with seq = 1 for kv cache mode.
                decoder_input_ids = input_ids[:,-1:]
                if config.use_mask:
                    decoder_attention_mask = torch.tensor([[1]]).to(device)
            else:
                # For decoder-only models, we need to use the entire input_ids as dummy inputs.
                # Passing the full input_ids lets us capture the behavior of the
                # context phase for models that treat it as a special case (like
                # OPT and BLOOM).
                # Passing past_key_values lets us capture the behavior of the
                # generation phase.
                # If a model requires that SeqLen == 1 to properly run the
                # generation phase, this may break.
                decoder_input_ids = input_ids
                if config.use_mask:
                    decoder_attention_mask = attention_mask
            decoder_outputs = decoder_with_lm_head.forward(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                image_embeds=image_embeds,
                use_cache=True,
                past_key_values=None
            ) # decoder output at t-1 step (logits, past_key_values from 0 to t-1)

            past_key_values = decoder_outputs[1]

            old_forward = decoder_with_lm_head.forward

            def _export_forward(input_ids, attention_mask, image_embeds, past_key_values):
                decoder_outputs = old_forward(input_ids, attention_mask, image_embeds=image_embeds, past_key_values=past_key_values, use_cache=True)
                past_key_values = ()
                past_key_values_output = decoder_outputs[1]
                for layer_past_states in past_key_values_output:
                    past_key_values = past_key_values + (layer_past_states[:2],)

                logits = decoder_outputs.logits
                if image_embeds is not None:
                    logits = logits.view(image_embeds.shape[0], logits.shape[1], logits.shape[2])

                return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

            decoder_with_lm_head.forward = _export_forward

            # Attention mask = curr_seq + past_seq(=curr_seq in export), so it needs to be expanded
            if config.use_mask:
                decoder_attention_mask = torch.cat((decoder_attention_mask, decoder_attention_mask), 1)

            torch.onnx.export(
                decoder_with_lm_head,
                (decoder_input_ids, decoder_attention_mask, image_embeds, past_key_values),
                output_fpath,
                do_constant_folding=CONSTANT_FOLDING,
                opset_version=OPSET,
                input_names=inputs.get_names(),
                output_names=outputs.get_names(),
                dynamic_axes={
                    **inputs.get_torch_dynamic_axis_encoding(),
                    **outputs.get_torch_dynamic_axis_encoding(),
                },
                training=TRAINING_MODE,
            )

        if network_metadata.precision.fp16:
            self.post_process_onnx(output_fpath)

        return self.onnx_class(model = output_fpath, network_metadata = network_metadata)

class Vision2SeqModelClass:
    """
    A class to track which class to use for each model type.
    """

    decoder_classes = {
        "torch": DecoderTorchFile,
        "onnx": DecoderONNXFile,
        "engine": DecoderTRTEngine
    }

    encoder_classes = {
        "torch": EncoderTorchFile,
        "onnx": EncoderONNXFile,
        "engine": EncoderTRTEngine
    }