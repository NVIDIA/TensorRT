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

import contextlib
import os
import time
from collections import OrderedDict

from polygraphy import constants, cuda, mod, util
from polygraphy.backend.base import BaseLoader, BaseRunner
from polygraphy.backend.trt import util as trt_util
from polygraphy.backend.trt.loader import BaseNetworkFromOnnx
from polygraphy.backend.trt.util import get_trt_logger
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER
from polygraphy.util.format import DataFormat, FormatManager

np = mod.lazy_import("numpy")
trt = mod.lazy_import("tensorrt")
uff = mod.lazy_import("uff")


class LoadUffFile(BaseLoader):
    def __init__(self, path, shapes, outputs):
        self.path = path
        self.shapes = shapes
        self.outputs = outputs

    def call_impl(self):
        input_names = list(self.shapes.keys())
        input_shapes = list(self.shapes.values())
        with open(self.path, "rb") as f:
            return f.read(), input_names, input_shapes, self.outputs


class ConvertToUff(BaseLoader):
    def __init__(self, tf_loader, save_uff=None, preprocessor=None):
        self.tf_loader = tf_loader
        self.uff_path = save_uff
        self.preprocessor = preprocessor

    def call_impl(self):
        """

        save_uff (bool): Whether to write the generated UFF and corresponding PBTXT files.
        """
        graph, output_names = self.tf_loader()
        output_names = [name.split(":")[0] for name in output_names]
        # GraphDefs don't have names, so we have to name it something generic.
        output_filename = None if not self.uff_path else "out.uff"

        # Generate the UFF model and get information about the input_buffers/output_buffers.
        uff_model, input_nodes, _ = uff.from_tensorflow(
            graph.as_graph_def(),
            return_graph_info=True,
            quiet=(G_LOGGER.module_severity.get(G_LOGGER.module_path(__file__)) > G_LOGGER.VERBOSE),
            debug_mode=(G_LOGGER.module_severity.get(G_LOGGER.module_path(__file__)) == G_LOGGER.EXTRA_VERBOSE),
            text=self.uff_path,
            save_preprocessed=self.uff_path,
            output_filename=output_filename,
            preprocessor=self.preprocessor,
        )

        input_names = [node.name for node in input_nodes]
        input_shapes = [tuple(int(dim.size) for dim in node.attr["shape"].shape.dim) for node in input_nodes]
        return uff_model, input_names, input_shapes, output_names


class LoadNetworkFromUff(BaseLoader):
    def __init__(self, uff_loader, uff_order=None):
        self.uff_loader = uff_loader
        self.uff_order = None
        if uff_order:
            self.uff_order = trt.UffInputOrder.NCHW if uff_order.lower() == "nchw" else trt.UffInputOrder.NHWC

    def call_impl(self):
        uff_model, input_names, input_shapes, output_names = self.uff_loader()

        builder = trt.Builder(get_trt_logger())
        network = builder.create_network()
        parser = trt.UffParser()
        # Input names should come from the converter, as a preprocessing script may have been applied to the frozen model.
        for name, shape in zip(input_names, input_shapes):
            # Default order is NCHW, only set to NHWC if we're reasonably certain that it is.
            input_order = self.uff_order
            if not self.uff_order:
                input_order = trt.UffInputOrder.NCHW
                if FormatManager.determine_format(shape) == DataFormat.NHWC:
                    input_order = trt.UffInputOrder.NHWC
            shape = shape[1:]
            G_LOGGER.verbose(f"Registering UFF input: {name} with shape: {shape} and input order: {input_order}")
            parser.register_input(name, shape, input_order)

        if output_names and output_names != constants.MARK_ALL:
            for name in output_names:
                G_LOGGER.verbose("Registering UFF output: " + str(name))
                parser.register_output(name)

        G_LOGGER.info(f"Parsing UFF model with inputs: {input_names} and outputs: {output_names}")
        success = parser.parse_buffer(uff_model, network)
        if not success:
            G_LOGGER.critical("Could not parse UFF correctly")
        return builder, network, parser, input_shapes[0][0]


class ParseNetworkFromOnnxLegacy(BaseNetworkFromOnnx):
    def __init__(self, onnx_loader):
        """
        Parses an ONNX model to create a trt.INetworkDefinition. This loader only supports the
        implicit batch version of the parser.

        Args:
            onnx_loader (Union[onnx.ModelProto, Callable() -> onnx.ModelProto]):
                    An ONNX model or a callable that returns one.
        """
        super().__init__(explicit_batch=False)
        self.onnx_loader = onnx_loader

    def call_impl(self):
        from polygraphy.backend.onnx import util as onnx_util

        with util.FreeOnException(super().call_impl()) as (builder, network, parser):
            onnx_model, _ = util.invoke_if_callable(self.onnx_loader)
            _, shape = list(onnx_util.get_input_metadata(onnx_model.graph).values())[0]

            success = parser.parse(onnx_model.SerializeToString())
            trt_util.check_onnx_parser_errors(parser, success)

            return builder, network, parser, shape[0]


class LoadNetworkFromCaffe:
    def __init__(self, deploy, model, outputs, batch_size=None, dtype=None):
        self.deploy = deploy
        self.model = model
        if not self.model:
            G_LOGGER.warning(
                "No model file provided for Caffe model, random weights will be used. To avoid this, "
                "please set the model paramater, or --model"
            )

        if not outputs:
            G_LOGGER.critical(
                f"Please set Caffe model outputs using the outputs parameter, or --trt-outputs. "
                "Note: To determine possible outputs, try running: tail -n50 {deploy}"
            )

        self.outputs = outputs
        self.dtype = util.default(dtype, trt.float32)
        self.batch_size = util.default(batch_size, 1)

    def __call__(self):
        builder = trt.Builder(get_trt_logger())
        network = builder.create_network()
        parser = trt.CaffeParser()

        parser.parse(deploy=self.deploy, model=self.model, network=network, dtype=self.dtype)

        if self.outputs and self.outputs != constants.MARK_ALL:
            trt_util.mark_outputs(network, self.outputs)

        return builder, network, parser, self.batch_size


def _input_metadata_from_network(network):
    input_metadata = TensorMetadata()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        input_metadata.add(name=tensor.name, dtype=np.dtype(trt.nptype(tensor.dtype)), shape=tensor.shape)
    return input_metadata


# Builds and tracks a single engine for a single network.
class TrtLegacyRunner(BaseRunner):
    """
    A runner that can perform inference on a single TensorRT engine.
    """

    # Simple helper data class that's a little nicer to use than a 2-tuple.
    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:" + str(self.host) + ", Device:" + str(self.device)

    def __init__(
        self,
        network_loader=None,
        max_workspace_size=None,
        max_batch_size=None,
        fp16=None,
        tf32=None,
        fp8=None,
        load_engine=None,
        save_engine=None,
        layerwise=False,
        plugins=[],
        name=None,
        int8=None,
        calibrator=None,
        use_dla=None,
        allow_gpu_fallback=None,
    ):
        """
        Creates a runner that manages a single TensorRT engine.


            network_loader (BaseModelLoader):
                    A loader that returns a TRT builder, network, parser and input shapes.
            max_workspace_size (int): The maximum workspace size.
            max_batch_size (int): The maximum batch size.
            fp16 (bool): Whether to run in fp16 mode
            fp8  (bool): Whether to run in fp8 mode            
            layerwise (bool): Whether to retrieve the outputs of every layer in the network.
            name (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
        """
        G_LOGGER.warning("TrtLegacyRunner is deprecated, and will be removed in a future release")
        # Load any user-supplied plugin libraries. This must happen before everything else, including engine deserialization.
        if plugins:
            import ctypes

            for plugin in plugins:
                path = os.path.abspath(plugin)
                G_LOGGER.info(f"Loading plugin library: {path}")
                ctypes.CDLL(path)

        # Choose a unique name for this runner.
        super().__init__(name=name, prefix="trt-legacy-runner")

        # Save parameters for activate and deactivate.
        self.network_loader = network_loader
        self.max_workspace_size = util.default(max_workspace_size, 1 << 24)
        self.fp16 = util.default(fp16, False)
        self.fp8  = util.default(fp8, False)
        self.tf32 = util.default(tf32, False)
        self.load_engine = load_engine

        self.engine_path = save_engine

        self.layerwise = layerwise
        self.max_batch_size = max_batch_size
        self.int8 = util.default(int8, False)
        self.calibrator = calibrator
        self.use_dla = use_dla
        self.allow_gpu_fallback = allow_gpu_fallback

    def activate_impl(self):
        """
        Parses command-line arguments and populates the following attributes:

        Attributes:
            engine (trt.ICudaEngine):
                    The engine tracked by this runner. The TrtLegacyRunner OWNS the engine it
                    manages, and therefore is responsible for it's destruction. Do not free the engine outside of the
                    runner, or it will result in a double free.
            context (trt.IExecutionContext): The context used for inference.
            input_buffers (Dict[str, TrtLegacyRunner.HostDeviceMem]):
                    A mapping of binding names to HostDeviceMem objects for input buffers.
            output_buffers (Dict[str, TrtLegacyRunner.HostDeviceMem]):
                    A mapping of binding names to HostDeviceMem objects for output buffers.
            bindings (List[int]): A list of device pointers for engine bindings.
            stream (cuda.Stream): The CUDA stream that this runner will use for inference.
        """
        # Only initialize GPU after this runner is activated.
        # Allocates all buffers required for an engine, i.e. host/device input_buffers/output_buffers.
        def allocate_buffers(engine):
            input_buffers = OrderedDict()
            output_buffers = OrderedDict()
            stream = cuda.Stream()
            G_LOGGER.verbose("Using batch size: " + str(engine.max_batch_size) + " during buffer allocation")
            for binding in engine:
                shape = (engine.max_batch_size,) + tuple(engine.get_binding_shape(binding))
                dtype = engine.get_binding_dtype(binding)

                device_mem = cuda.DeviceArray(shape=shape, dtype=trt.nptype(dtype))
                G_LOGGER.extra_verbose(f"Tensor: {binding:35} | Allocated: {device_mem}")

                if engine.binding_is_input(binding):
                    input_buffers[binding] = TrtLegacyRunner.HostDeviceMem(None, device_mem)
                else:
                    host_mem = np.empty(shape=shape, dtype=trt.nptype(dtype))
                    output_buffers[binding] = TrtLegacyRunner.HostDeviceMem(host_mem, device_mem)
            return input_buffers, output_buffers, stream

        # Always try reading the engine first, or, failing that, build it.
        if self.load_engine:
            with open(self.load_engine, "rb") as f, trt.Runtime(get_trt_logger()) as runtime:
                G_LOGGER.info(f"Reading engine from {self.load_engine}")
                self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            trt.init_libnvinfer_plugins(get_trt_logger(), "")
            builder, network, parser, model_batch_size = self.network_loader()
            with builder, network, parser, builder.create_builder_config() as config, contextlib.ExitStack() as stack:
                if not network:
                    G_LOGGER.critical("Invalid network")
                G_LOGGER.super_verbose(lambda: trt_util.str_from_network(network) or "Finished logging network")

                builder.max_batch_size = int(self.max_batch_size or model_batch_size or 1)

                config.max_workspace_size = int(self.max_workspace_size)

                if not self.tf32:
                    with contextlib.suppress(AttributeError):
                        config.clear_flag(trt.BuilderFlag.TF32)
                if self.fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                if self.fp8:
                    config.set_flag(trt.BuilderFlag.FP8)

                if self.int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    input_metadata = _input_metadata_from_network(network)
                    with contextlib.suppress(AttributeError):  # Polygraphy calibrator has a reset method
                        self.calibrator.set_input_metadata(input_metadata)
                        self.calibrator.reset()
                    config.int8_calibrator = self.calibrator

                if self.use_dla:
                    config.default_device_type = trt.DeviceType.DLA
                    config.DLA_core = 0

                if self.allow_gpu_fallback:
                    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

                if self.layerwise:
                    trt_util.mark_layerwise(network)

                G_LOGGER.info(
                    f"Building engine: max workspace size={config.max_workspace_size} bytes, max batch size={builder.max_batch_size}, "
                    f"fp16={self.fp16}, tf32={self.tf32}, int8={self.int8}, fp8={self.fp8}"
                )
                self.engine = builder.build_engine(network, config)

        if not self.engine:
            G_LOGGER.critical("Invalid Engine. Please ensure the engine was built correctly")

        if self.engine_path:
            with open(self.engine_path, "wb") as f:
                G_LOGGER.info(f"Writing engine to {self.engine_path}")
                f.write(self.engine.serialize())

        self.context = self.engine.create_execution_context()
        self.input_buffers, self.output_buffers, self.stream = allocate_buffers(self.engine)

    def get_input_metadata_impl(self):
        inputs = TensorMetadata()

        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                # Always prepend a dynamic batch dimension
                inputs.add(
                    binding,
                    trt.nptype(self.engine.get_binding_dtype(binding)),
                    [-1] + list(self.engine.get_binding_shape(binding)),
                )
        return inputs

    def deactivate_impl(self):
        # Destroy the engine and context.
        with self.engine, self.context:
            pass
        [inp.device.free() for inp in self.input_buffers.values()]
        [out.device.free() for out in self.output_buffers.values()]
        self.stream.free()

        del (self.engine, self.context, self.input_buffers, self.output_buffers, self.stream)

    def infer_impl(self, feed_dict):
        start = time.time()

        for name, buffer in feed_dict.items():
            self.input_buffers[name].device.resize(buffer.shape)
            buffer = util.make_contiguous(buffer)
            self.input_buffers[name].device.copy_from(buffer, self.stream)

        # We will not run with smaller batch sizes than whatever the builder chose.
        bindings = [buf.device.ptr for buf in self.input_buffers.values()] + [
            buf.device.ptr for buf in self.output_buffers.values()
        ]
        status = self.context.execute_async(
            batch_size=self.context.engine.max_batch_size, bindings=bindings, stream_handle=self.stream.ptr
        )
        if not status:
            G_LOGGER.critical("Model execution failed. Please see the log messages above for details")

        for out in self.output_buffers.values():
            out.host = util.resize_buffer(out.host, out.device.shape)
            out.device.copy_to(out.host, self.stream)

        self.stream.synchronize()
        end = time.time()

        out_dict = OrderedDict()
        for (name, out) in self.output_buffers.items():
            out_dict[name] = out.host
        self.inference_time = end - start
        return out_dict
