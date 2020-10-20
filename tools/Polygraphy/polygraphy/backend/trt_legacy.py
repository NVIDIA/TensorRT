#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#

from polygraphy.backend.base import BaseRunner, BaseLoadModel
from polygraphy.backend.trt.loader import BaseNetworkFromOnnx
from polygraphy.util.format import DataFormat, FormatManager
from polygraphy.common import TensorMetadata, constants
from polygraphy.backend.trt import util as trt_util
from polygraphy.backend.trt.util import TRT_LOGGER
from polygraphy.logger.logger import G_LOGGER
from polygraphy.util import misc, cuda


from collections import OrderedDict
import contextlib
import time
import os

import tensorrt as trt
import numpy as np

misc.log_module_info(trt)

class LoadUffFile(BaseLoadModel):
    def __init__(self, path, shapes, outputs):
        self.path = path
        self.shapes = shapes
        self.outputs = outputs

    def __call__(self):
        input_names = list(self.shapes.keys())
        input_shapes = list(self.shapes.values())
        with open(self.path, "rb") as f:
            return f.read(), input_names, input_shapes, self.outputs


class ConvertToUff(BaseLoadModel):
    def __init__(self, tf_loader, save_uff=None, preprocessor=None):
        self.tf_loader = tf_loader
        self.uff_path = save_uff
        self.preprocessor = preprocessor

    def __call__(self):
        """

            save_uff (bool): Whether to write the generated UFF and corresponding PBTXT files.
        """
        from polygraphy.backend.tf import util as tf_util
        import uff
        misc.log_module_info(uff)

        graph, output_names = self.tf_loader()
        output_names = [name.split(":")[0] for name in output_names]
        # GraphDefs don't have names, so we have to name it something generic.
        output_filename = None if not self.uff_path else "out.uff"

        # Generate the UFF model and get information about the input_buffers/output_buffers.
        uff_model, input_nodes, _ = uff.from_tensorflow(graph.as_graph_def(), return_graph_info=True,
                                                        quiet=(G_LOGGER.severity > G_LOGGER.VERBOSE),
                                                        debug_mode=(G_LOGGER.severity == G_LOGGER.EXTRA_VERBOSE), text=self.uff_path,
                                                        save_preprocessed=self.uff_path, output_filename=output_filename,
                                                        preprocessor=self.preprocessor)

        input_names = [node.name for node in input_nodes]
        input_shapes = [tuple(int(dim.size) for dim in node.attr["shape"].shape.dim) for node in input_nodes]
        return uff_model, input_names, input_shapes, output_names


class LoadNetworkFromUff(BaseLoadModel):
    def __init__(self, uff_loader, uff_order=None):
        self.uff_loader = uff_loader
        self.uff_order = None
        if uff_order:
            self.uff_order = trt.UffInputOrder.NCHW if uff_order.lower() == "nchw" else trt.UffInputOrder.NHWC

    def __call__(self):
        uff_model, input_names, input_shapes, output_names = self.uff_loader()

        builder = trt.Builder(TRT_LOGGER)
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
            G_LOGGER.verbose("Registering UFF input: {:} with shape: {:} and input order: {:}".format(name, shape, input_order))
            parser.register_input(name, shape, input_order)

        if output_names and output_names != constants.MARK_ALL:
            for name in output_names:
                G_LOGGER.verbose("Registering UFF output: " + str(name))
                parser.register_output(name)

        G_LOGGER.info("Parsing UFF model with inputs: {:} and outputs: {:}".format(input_names, output_names))
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
            onnx_loader (Callable() -> onnx.ModelProto): A loader that can supply an ONNX model.
        """
        super().__init__(explicit_precision=False, explicit_batch=False)
        self.onnx_loader = onnx_loader


    def __call__(self):
        from polygraphy.backend.onnx import util as onnx_util

        builder, network, parser = super().__call__()
        onnx_model, _ = misc.try_call(self.onnx_loader)
        dtype, shape = list(onnx_util.get_input_metadata(onnx_model.graph).values())[0]

        parser.parse(onnx_model.SerializeToString())
        trt_util.check_onnx_parser_errors(parser)

        return builder, network, parser, shape[0]


class LoadNetworkFromCaffe(BaseLoadModel):
    def __init__(self, deploy, model, outputs, batch_size=None, dtype=None):
        self.deploy = deploy
        self.model = model
        if not self.model:
            G_LOGGER.warning("No model file provided for Caffe model, random weights will be used. To avoid this, "
                             "please set the model paramater, or --model")

        if not outputs:
            G_LOGGER.critical("Please set Caffe model outputs using the outputs parameter, or --trt-outputs. "
                              "Note: To determine possible outputs, try running: tail -n50 {:}".format(deploy))

        self.outputs = outputs
        self.dtype = misc.default_value(dtype, trt.float32)
        self.batch_size = misc.default_value(batch_size, 1)

    def __call__(self):
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network()
        parser = trt.CaffeParser()

        model_tensors = parser.parse(deploy=self.deploy, model=self.model, network=network, dtype=self.dtype)

        if self.outputs and self.outputs != constants.MARK_ALL:
            for output in self.outputs:
                network.mark_output(model_tensors.find(output))

        return builder, network, parser, self.batch_size


# Builds and tracks a single engine for a single network.
class TrtLegacyRunner(BaseRunner):
    """
    A runner that can perform inference on a single TensorRT engine.
    """
    # Simple helper data class that's a little nicer to use than a 2-tuple.
    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:" + str(self.host) + ", Device:" + str(self.device)

    def __init__(self, network_loader=None, max_workspace_size=None, max_batch_size=None, fp16=None,
                 tf32=None, load_engine=None, save_engine=None, layerwise=False, plugins=[], name=None):
        """
        Creates a runner that manages a single TensorRT engine.


            network_loader (BaseModelLoader):
                    A loader that returns a TRT builder, network, parser and input shapes.
            max_workspace_size (int): The maximum workspace size.
            max_batch_size (int): The maximum batch size.
            fp16 (bool): Whether to run in fp16 mode
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
                G_LOGGER.info("Loading plugin library: {:}".format(path))
                ctypes.CDLL(path)

        # Choose a unique name for this runner.
        super().__init__(name=name, prefix="trt-legacy-runner")

        # Save parameters for activate and deactivate.
        self.network_loader = network_loader
        self.max_workspace_size = misc.default_value(max_workspace_size, 1<<24)
        self.fp16 = misc.default_value(fp16, False)
        self.tf32 = misc.default_value(tf32, False)
        self.load_engine = load_engine

        self.engine_path = save_engine

        self.layerwise = layerwise
        self.max_batch_size = max_batch_size


    def activate_impl(self):
        """
        Vars:
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
            bindings = []
            stream = cuda.Stream()
            G_LOGGER.verbose("Using batch size: " + str(engine.max_batch_size) + " during buffer allocation")
            for binding in engine:
                shape = (engine.max_batch_size, ) + tuple(engine.get_binding_shape(binding))
                dtype = engine.get_binding_dtype(binding)

                device_mem = cuda.DeviceBuffer(shape=shape, dtype=trt.nptype(dtype))
                G_LOGGER.extra_verbose("Tensor: "
                               "{:40} | Allocated: {:}".format(binding, device_mem))

                if engine.binding_is_input(binding):
                    input_buffers[binding] = TrtLegacyRunner.HostDeviceMem(None, device_mem)
                else:
                    host_mem = np.empty(shape=shape, dtype=trt.nptype(dtype))
                    output_buffers[binding] = TrtLegacyRunner.HostDeviceMem(host_mem, device_mem)
            return input_buffers, output_buffers, stream

        # Always try reading the engine first, or, failing that, build it.
        if self.load_engine:
            with open(self.load_engine, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                G_LOGGER.info("Reading engine from {:}".format(self.load_engine))
                self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            trt.init_libnvinfer_plugins(TRT_LOGGER, "")
            builder, network, parser, model_batch_size = self.network_loader()
            with builder, network, parser:
                builder.max_batch_size = int(self.max_batch_size or model_batch_size or 1)

                config = builder.create_builder_config()
                config.max_workspace_size = int(self.max_workspace_size)

                if not self.tf32:
                    with contextlib.suppress(AttributeError): config.clear_flag(trt.BuilderFlag.TF32)
                if self.fp16:
                    config.flags = 1 << int(trt.BuilderFlag.FP16)

                if not network:
                    G_LOGGER.critical("Invalid network")
                G_LOGGER.super_verbose(lambda: trt_util.str_from_network(network) or "Finished logging network")


                if self.layerwise:
                    # In layerwise mode, every layer becomes an output.
                    G_LOGGER.info("Running in layerwise mode. Marking {:} layers as outputs".format(network.num_layers))
                    for layer in network:
                        for index in range(layer.num_outputs):
                            out = layer.get_output(index)
                            if not out.is_network_output:
                                network.mark_output(out)

                G_LOGGER.info("Building engine: max workspace size={:} bytes, max batch size={:}, fp16={:}, "
                              "tf32={:}".format(builder.max_workspace_size, builder.max_batch_size, self.fp16, self.tf32))
                self.engine = builder.build_engine(network, config)


        if not self.engine:
            G_LOGGER.critical("Invalid Engine. Please ensure the engine was built correctly")

        if self.engine_path:
            with open(self.engine_path, "wb") as f:
                G_LOGGER.info("Writing engine to {:}".format(self.engine_path))
                f.write(self.engine.serialize())

        self.context = self.engine.create_execution_context()
        self.input_buffers, self.output_buffers, self.stream = allocate_buffers(self.engine)


    def get_input_metadata(self):
        inputs = TensorMetadata()

        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                # Always prepend a dynamic batch dimension
                inputs[binding] = (trt.nptype(self.engine.get_binding_dtype(binding)), [-1] + list(self.engine.get_binding_shape(binding)))
        return inputs


    def deactivate_impl(self):
        # Destroy the engine and context.
        with self.engine, self.context:
            pass
        [inp.device.free() for inp in self.input_buffers.values()]
        [out.device.free() for out in self.output_buffers.values()]
        self.stream.free()


    def infer(self, feed_dict):
        start = time.time()
        [self.input_buffers[name].device.copy_from(buffer, self.stream) for name, buffer in feed_dict.items()]
        # We will not run with smaller batch sizes than whatever the builder chose.
        bindings = [buf.device.address() for buf in self.input_buffers.values()] + [buf.device.address() for buf in self.output_buffers.values()]
        status = self.context.execute_async(batch_size=self.context.engine.max_batch_size, bindings=bindings,
                                            stream_handle=self.stream.address())
        if not status:
            G_LOGGER.critical("Model execution failed. Please see the log messages above for details")

        for out in self.output_buffers.values():
            out.host = out.device.copy_to(out.host, self.stream)

        self.stream.synchronize()
        end = time.time()

        out_dict = OrderedDict()
        for (name, out) in self.output_buffers.items():
            out_dict[name] = out.host
        self.inference_time = end - start
        return out_dict
