#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

"""Utilities related to Polygraphy"""

from typing import List

# polygraphy
from polygraphy.backend.trt import engine_from_bytes, TrtRunner
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx, OnnxrtRunner
from polygraphy.backend.common import bytes_from_path
from polygraphy.logger import G_LOGGER as PG_LOGGER

# tensorrt
import tensorrt as trt

# ONNX
import onnx
import onnx_graphsurgeon as gs

# numpy
import numpy as np

# NNDF
from NNDF.networks import NetworkMetadata
from NNDF.models import TRTEngineFile
from NNDF.logger import G_LOGGER

# Helper Functions
def clamp_weights_onnx(onnx_input_fpath: str, onnx_output_fpath: str, min: float, max: float, ignore_nodes: List = None):
    """
    Clamps given onnx model to targeted upper and lower bounds.
    """

    graph = gs.import_onnx(onnx.load(onnx_input_fpath))
    if ignore_nodes is None:
        ignore_nodes = {}
    else:
        ignore_nodes = {k: True for k in ignore_nodes}

    for tensor in graph.tensors().values():
        if tensor.name in ignore_nodes or isinstance(tensor, gs.ir.tensor.Variable):
            continue

        np.clip(tensor.values, min, max, out=tensor.values)

    for tensor in graph.nodes:
        node_attr = tensor.attrs.get("value", None)
        if tensor.name in ignore_nodes:
            continue

        if node_attr is not None:
            np.clip(node_attr.values, min, max, out=node_attr.values)

    model = gs.export_onnx(graph)
    onnx.save(model, onnx_output_fpath, save_as_external_data=True)


def clamp_weights_onnx_to_fp16_bounds(onnx_input_fpath: str, onnx_output_fpath: str, ignore_nodes: List = None):
    upper_bound = 65504
    return clamp_weights_onnx(onnx_input_fpath, onnx_output_fpath, -upper_bound, upper_bound, ignore_nodes)


def move_t5_cast_op(onnx_input_fpath: str, onnx_output_fpath: str):
    """
    T5 encoder and decoder have cast ops after residual add operation.
    Moving the cast operation before add helps with FP16 accuracy as addition operation
    can cause overflow in FP16.
    """

    graph = gs.import_onnx(onnx.load(onnx_input_fpath))
    cast_nodes = [node for node in graph.nodes if node.op == "Cast"]
    for n in cast_nodes:
        # Cast appears at the output of add and feeds into a Pow op.
        if n.i().op == "Add":
            found_pow = False
            for o in n.outputs:
                for o1 in o.outputs:
                    if o1.op == "Pow":
                        found_pow = True

            if found_pow:
                n.i().outputs = n.outputs
                n.outputs.clear()

    graph.cleanup().toposort()
    add_nodes = [node for node in graph.nodes if node.op == "Add"]
    for n in add_nodes:
        if n.o().op == "Pow":
            add_inputs = n.inputs
            outs = []
            for i in  add_inputs:
                identity_out = gs.Variable("identity_out" + i.name, dtype=np.float32)
                new_cast = gs.Node(op="Cast", inputs=[i], outputs=[identity_out], attrs={"to": 1})
                outs.append(identity_out)
                graph.nodes.append(new_cast)
            n.inputs = outs

    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    onnx.save(model, onnx_output_fpath, save_as_external_data=True)

# Helper Classes
class TRTNativeRunner:
    """TRTNativeRunner avoids the high overheads with Polygraphy runner providing performance comparable to C++ implementation."""
    def __init__(self, trt_engine_file: TRTEngineFile, network_metadata: NetworkMetadata):
        self.trt_engine_file = trt_engine_file
        trt_logger = trt.Logger(trt.Logger.VERBOSE if G_LOGGER.root.level == G_LOGGER.DEBUG else trt.Logger.WARNING)

        G_LOGGER.info("Reading and loading engine file {} using trt native runner.".format(self.trt_engine_file.fpath))
        with open(self.trt_engine_file.fpath, "rb") as f:
            self.trt_runtime = trt.Runtime(trt_logger)
            self.trt_engine = self.trt_runtime.deserialize_cuda_engine(f.read())
            self.trt_context = self.trt_engine.create_execution_context()

        # By default set optimization profile to 0
        self.profile_idx = 0

        # Other metadata required by the profile
        self._num_bindings_per_profile = self.trt_engine.num_bindings // self.trt_engine.num_optimization_profiles
        G_LOGGER.debug("Number of profiles detected in engine: {}".format(self._num_bindings_per_profile))

    def release(self):
        pass

    def get_optimization_profile(self, batch_size, sequence_length):
        """Provided helper function to obtain a profile optimization."""
        # Select an optimization profile
        # inspired by demo/BERT/inference.py script
        selected_profile_idx = None
        for idx in range(self.trt_engine.num_optimization_profiles):
            profile_shape = self.trt_engine.get_profile_shape(profile_index=idx, binding=idx * self._num_bindings_per_profile)

            if profile_shape[0][0] <= batch_size and profile_shape[2][0] >= batch_size \
               and profile_shape[0][1] <=  sequence_length and profile_shape[2][1] >= sequence_length:
                G_LOGGER.debug("Selected profile: {}".format(profile_shape))
                selected_profile_idx = idx
                break

        if selected_profile_idx == -1:
            raise RuntimeError("Could not find any profile that matches batch_size={}, sequence_length={}".format(batch_size, sequence_length))

        return selected_profile_idx

    def __call__(self, *args, **kwargs):
        self.trt_context.active_optimization_profile = self.profile_idx
        return self.forward(*args, **kwargs)

class PolygraphyOnnxRunner:
    def __init__(self, onnx_fpath: str, network_metadata: NetworkMetadata):
        self.network_metadata = network_metadata
        self.trt_session = SessionFromOnnx(onnx_fpath)
        self.trt_context = OnnxrtRunner(self.trt_session)
        self.trt_context.activate()

    def __call__(self, *args, **kwargs):
        # hook polygraphy verbosity for inference
        g_logger_verbosity = (
            G_LOGGER.EXTRA_VERBOSE
            if G_LOGGER.root.level == G_LOGGER.DEBUG
            else G_LOGGER.WARNING
        )
        with PG_LOGGER.verbosity(g_logger_verbosity):
            return self.forward(*args, **kwargs)

    def release(self):
        self.trt_context.deactivate()

class TRTPolygraphyRunner:
    """
    TRT implemented network interface that can be used to measure inference time.
    Easier to use but harder to utilize. Recommend using TRTNativeRunner for better performance.
    """

    def __init__(self, engine_fpath: str, network_metadata: NetworkMetadata):
        self.network_metadata = network_metadata

        self.trt_engine = engine_from_bytes(bytes_from_path(engine_fpath))
        self.trt_context = TrtRunner(self.trt_engine.create_execution_context())
        self.trt_context.activate()

    def __call__(self, *args, **kwargs):
        # hook polygraphy verbosity for inference
        g_logger_verbosity = (
            G_LOGGER.EXTRA_VERBOSE
            if G_LOGGER.root.level == G_LOGGER.DEBUG
            else G_LOGGER.WARNING
        )

        with PG_LOGGER.verbosity(g_logger_verbosity):
            return self.forward(*args, **kwargs)

    def release(self):
        self.trt_context.deactivate()
