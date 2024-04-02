#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ctypes
from polygraphy import mod
from polygraphy.logger import G_LOGGER
from polygraphy.tools.sparse import SparsityPruner
from polygraphy.tools.surgeon.subtool.base import BaseSurgeonSubtool
from polygraphy.tools.args import ModelArgs, OnnxLoadArgs, OnnxSaveArgs

np = mod.lazy_import("numpy")
onnx = mod.lazy_import("onnx")
torch = mod.lazy_import("torch")

class WeightReconstructor(BaseSurgeonSubtool):
    """
    Reconstruct proxy weights in the Stripped ONNX model
    """
    def __init__(self):
        super().__init__("weight-reconstruct")
    
    def show_start_end_logging_impl(self, args):
        return True

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=True, input_shapes_opt_name=False, required_model_type="onnx"),
            OnnxLoadArgs(allow_shape_inference=False, outputs_opt_prefix=False, allow_from_tf=False),
            OnnxSaveArgs(allow_shape_inference=False, output_opt_required=True),
        ]
    
    def run_impl_surgeon(self, args):
        def reconstruct_weights(model):
            G_LOGGER.start(f"Beginning weight reconstruction...")
            # Skip Sparsity Pruning of weights not marked as "SPARSE_2_4"
            skip_weight_sparsify = set()
            num_reconstructed = 0
            for initializer in model.graph.initializer:
                doc_string = initializer.doc_string

                # If not marked as weightless, leave initializer untouched
                if "TRT_WEIGHTLESS" not in doc_string:
                    skip_weight_sparsify.add(initializer.name)
                    continue
                _, sparse_str = doc_string.split('/')

                # If not sparse, add to skip list
                if not sparse_str:
                    skip_weight_sparsify.add(initializer.name)

                weight_dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
                weight_shape = tuple(initializer.dims)
                proxy_weight_tensor = np.random.randn(*weight_shape).astype(weight_dtype)
                proxy_weight_bytes = proxy_weight_tensor.data.tobytes()
                if initializer.data_type == onnx.TensorProto.BFLOAT16:
                    proxy_weight_tensor = torch.from_numpy(proxy_weight_tensor).to(torch.bfloat16)
                    proxy_weight_bytes = bytes((ctypes.c_byte * proxy_weight_tensor.numel()
                        * proxy_weight_tensor.element_size()).from_address(proxy_weight_tensor.untyped_storage().data_ptr()))
                assert weight_shape == proxy_weight_tensor.shape
                assert initializer.raw_data == b""

                G_LOGGER.verbose(f"Reconstructing weights for the {initializer.name} initializer")
                num_reconstructed += 1
                initializer.raw_data = proxy_weight_bytes

            # Call Sparsity Pruner tool to convert selected weights to sparse weights
            G_LOGGER.info("Calling Sparsity Pruner to prune selected weights")
            sparsity_pruner = SparsityPruner(model)
            model = sparsity_pruner.prune(weights_skip=skip_weight_sparsify)
            G_LOGGER.finish(f"Finished reconstructing {num_reconstructed} weights")
            return model

        model = super().load_model()
        reconstructed_model = reconstruct_weights(model)
        super().save_model(reconstructed_model)