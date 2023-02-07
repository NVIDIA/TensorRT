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
import time
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.backend.base import BaseRunner

torch = mod.lazy_import("torch")


@mod.export()
class PytRunner(BaseRunner):
    """
    Runs inference using PyTorch.
    """

    def __init__(self, model, input_metadata, output_names, name=None):
        """
        Args:
            model (Union[torch.nn.Module, Callable() -> torch.nn.Module]):
                    A torch.nn.Module or subclass or a callable that returns one.
            input_metadata (TensorMetadata): Mapping of input names to their data types and shapes.
            output_names (List[str]):
                    A list of output names of the model. This information is used by the
                    Comparator to determine which outputs to compare.


            name (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
        """
        super().__init__(name=name, prefix="pytorch-runner")
        self._model = model
        self.input_metadata = input_metadata
        self.output_names = output_names

    def activate_impl(self):
        self.model, _ = util.invoke_if_callable(self._model)
        self.model.eval()

    def get_input_metadata_impl(self):
        return self.input_metadata

    def infer_impl(self, feed_dict):
        with torch.no_grad():
            inputs = [
                torch.from_numpy(val.astype(dtype)).cuda()
                for (val, (dtype, _)) in zip(feed_dict.values(), self.input_metadata.values())
            ]
            start = time.time()
            outputs = self.model(*inputs)
            end = time.time()

        out_dict = OrderedDict()
        for name, output in zip(self.output_names, outputs):
            out_dict[name] = output.cpu().numpy()
        return out_dict, end - start

    def deactivate_impl(self):
        del self.model
