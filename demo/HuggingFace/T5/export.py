"""
Contains logic that captures T5 HuggingFace models into ONNX models.
Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py
"""

# std
from itertools import islice

# tensorrt
import tensorrt as trt

# polygraphy
from polygraphy.backend.trt import Profile

# torch
import torch
from torch.nn import Module

# huggingface
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput

# TRT-HuggingFace
from T5.T5ModelConfig import T5ModelTRTConfig
from NNDF.tensorrt_utils import clamp_weights_onnx_to_fp16_bounds
from NNDF.networks import NetworkMetadata
from NNDF.logger import G_LOGGER
from NNDF.models import (
    TRTEngineFile,
    TorchModelFile,
    ONNXModelFile,
    ModelFileConverter,
)

def add_extra_fp32(network_definition):
    def window(seq, n=2):
        "Returns a sliding window (of width n) over data from the iterable"
        "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    indices = list(range(0, network_definition[1].num_layers))
    for i, i_1, i_2, i_3, i_4, i_5 in window(indices, 6):
        l = network_definition[1].get_layer(i)
        l_1 = network_definition[1].get_layer(i_1)
        l_2 = network_definition[1].get_layer(i_2)
        l_3 = network_definition[1].get_layer(i_3)
        l_4 = network_definition[1].get_layer(i_4)
        l_5 = network_definition[1].get_layer(i_5)

        if not all([l.get_output(k).is_execution_tensor for k in range(l.num_outputs)]):
            continue

        if l.get_output_type(0) != trt.float32:
            continue

        if l.type == trt.LayerType.ELEMENTWISE and \
           l_1.type == trt.LayerType.REDUCE and \
           l_2.type == trt.LayerType.CONSTANT and \
           l_4.type == trt.LayerType.ELEMENTWISE and \
           l_5.type == trt.LayerType.UNARY:

            l.__class__ = getattr(trt, "IElementWiseLayer")
            if l.op == trt.ElementWiseOperation.POW:
                l.precision = trt.float32
                l.set_output_type(0, trt.float32)

            l_1.precision = trt.float32
            l_1.set_output_type(0, trt.float32)

            l_4.__class__ = getattr(trt, "IElementWiseLayer")
            if l_4.op == trt.ElementWiseOperation.SUM:
                l_4.precision = trt.float32
                l_4.set_output_type(0, trt.float32)

            l_5.__class__ = getattr(trt, "IUnaryLayer")
            if l_5.op == trt.UnaryOperation.SQRT:
                l_5.precision = trt.float32
                l_5.set_output_type(0, trt.float32)

    return network_definition

# Torch File Encoding #
class T5DecoderTorchFile(TorchModelFile):
    class TorchModule(Module, GenerationMixin):
        """
        A simplied definition of T5 Decoder without support for loss.
        Decoder with lm-head attached.
        """

        def __init__(self, decoder, lm_head, config):
            super().__init__()
            self.decoder = decoder
            self.lm_head = lm_head
            self.config = config

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {
                "input_ids": input_ids,
                "encoder_hidden_states": kwargs["encoder_hidden_states"],
            }

        def forward(self, input_ids, encoder_hidden_states, **kwargs):
            decoder_outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs
            )

            # self.config.d_model ** -0.5 for rescaling output on vocab.
            # as seen in https://huggingface.co/transformers/_modules/transformers/models/t5/modeling_t5.html#T5ForConditionalGeneration
            sequence_output = decoder_outputs[0] * self.config.d_model ** -0.5
            logits = self.lm_head(sequence_output)
            if not kwargs.get("return_dict", False):
                return (logits,) + decoder_outputs[1:]

            return Seq2SeqLMOutput(logits=logits)

    def __init__(self, model, network_metadata):
        super().__init__(model, T5DecoderConverter, network_metadata)


class T5EncoderTorchFile(TorchModelFile):
    """Creation of a class to output only the last hidden state from the encoder."""

    class TorchModule(Module, GenerationMixin):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, *input, **kwargs):
            return self.encoder(*input, **kwargs)[0]

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def __init__(self, model, network_metadata):
        super().__init__(model, T5EncoderConverter, network_metadata)


# ONNX File Encoding #
class T5EncoderONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, T5EncoderConverter, network_metadata)


class T5DecoderONNXFile(ONNXModelFile):
    def __init__(self, model, network_metadata):
        super().__init__(model, T5DecoderConverter, network_metadata)


# TRT Engine File Encoding #
class T5DecoderTRTEngine(TRTEngineFile):
    DEFAULT_TRT_WORKSPACE_MB = 3072

    def __init__(self, model, network_metadata):
        super().__init__(model, T5DecoderConverter, network_metadata)

    def get_network_definition(self, network_definition):
        return add_extra_fp32(network_definition)

    def get_dynamic_shape_profiles(self):
        max_sequence_length = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[
            self.network_metadata.variant
        ]
        profile = Profile()
        profile.add(
            "input_ids",
            min=(1, 1),
            opt=(1, max_sequence_length // 2),
            max=(1, max_sequence_length),
        )
        profile.add(
            "encoder_hidden_states",
            min=(1, 1, max_sequence_length),
            opt=(1, max_sequence_length // 2, max_sequence_length),
            max=(1, max_sequence_length, max_sequence_length),
        )
        return [profile]

    def use_strict_types(self):
        return self.network_metadata.precision.fp16


class T5EncoderTRTEngine(TRTEngineFile):
    DEFAULT_TRT_WORKSPACE_MB = 2048

    def __init__(self, model, network_metadata):
        super().__init__(model, T5EncoderConverter, network_metadata)

    def get_network_definition(self, network_definition):
        return add_extra_fp32(network_definition)

    def get_dynamic_shape_profiles(self):
        max_sequence_length = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[
            self.network_metadata.variant
        ]
        return [
            Profile().add(
                "input_ids",
                min=(1, 1),
                opt=(1, max_sequence_length // 2),
                max=(1, max_sequence_length),
            )
        ]

    def use_strict_types(self):
        return self.network_metadata.precision.fp16

# Converters #
class T5DecoderConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(T5DecoderTorchFile, T5DecoderONNXFile, T5DecoderTRTEngine)

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata
    ):
        """
        Exports a given huggingface T5 to decoder architecture only.
        Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            T5DecoderONNXFile: ONNX decoder object.
        """

        input_ids = torch.tensor([[42] * 10])
        # Exporting the decoder requires a basic instance of the encoder
        # Create one temporarily
        simplified_encoder = T5EncoderTorchFile.TorchModule(model.encoder)
        # Exports to ONNX
        decoder_with_lm_head = T5DecoderTorchFile.TorchModule(
            model.decoder, model.lm_head, model.config
        )

        # This code allows for huggingface compatible torch class to use onnx exporter
        old_forward = decoder_with_lm_head.forward
        def _export_forward(*args, **kwargs):
            result = old_forward(*args, **kwargs)
            return result[0]

        decoder_with_lm_head.forward = _export_forward

        inputs = T5ModelTRTConfig.get_input_dims(network_metadata)["decoder"]
        outputs = T5ModelTRTConfig.get_output_dims(network_metadata)["decoder"]

        torch.onnx.export(
            decoder_with_lm_head,
            (input_ids, simplified_encoder(input_ids)),
            output_fpath,
            export_params=True,
            opset_version=12,
            input_names=inputs.get_names(),
            output_names=outputs.get_names(),
            dynamic_axes={
                **inputs.get_torch_dynamic_axis_encoding(),
                **outputs.get_torch_dynamic_axis_encoding(),
            },
            training=False,
            use_external_data_format=True
        )

        if network_metadata.precision.fp16:
            G_LOGGER.debug("Clamping FP16 weights for T5")
            clamp_weights_onnx_to_fp16_bounds(output_fpath, output_fpath)

        return T5DecoderONNXFile(output_fpath, network_metadata)


class T5EncoderConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(T5EncoderTorchFile, T5EncoderONNXFile, T5EncoderTRTEngine)

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata
    ):
        """
        Exports a given huggingface T5 to encoder architecture only.
        Inspired by https://github.com/onnx/models/blob/master/text/machine_comprehension/t5/dependencies/T5-export.py

        Args:
            output_prefix (str): Path to the onnx file
            model (torch.Model): Model loaded torch class

        Returns:
            Tuple[str]: Names of generated models
        """
        input_ids = torch.tensor([[42] * 10])
        simplified_encoder = T5EncoderTorchFile.TorchModule(model.encoder)
        inputs = T5ModelTRTConfig.get_input_dims(network_metadata)["encoder"]
        outputs = T5ModelTRTConfig.get_output_dims(network_metadata)["encoder"]

        # Exports to ONNX
        torch.onnx._export(
            simplified_encoder,
            input_ids,
            output_fpath,
            export_params=True,
            opset_version=12,
            input_names=inputs.get_names(),
            output_names=outputs.get_names(),
            dynamic_axes={
                **inputs.get_torch_dynamic_axis_encoding(),
                **outputs.get_torch_dynamic_axis_encoding(),
            },
            training=False,
            use_external_data_format=True
        )

        if network_metadata.precision.fp16:
            G_LOGGER.debug("Clamping FP16 weights for T5")
            clamp_weights_onnx_to_fp16_bounds(output_fpath, output_fpath)

        return T5EncoderONNXFile(output_fpath, network_metadata)
