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
import torch
from polygraphy.backend.base import BaseRunner
from polygraphy.util import misc


class PytRunner(BaseRunner):
    def __init__(self, model, input_metadata, output_names, name=None):
        """
        Args:
            model (Callable() -> torch.nn.Module):
                    A model loader that returns a torch.nn.Module or subclass.
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
        self.model, _ = misc.try_call(self._model)
        self.model.eval()


    def infer(self, feed_dict):
        with torch.no_grad():
            inputs = [torch.from_numpy(val.astype(dtype)).cuda() for (val, (dtype, _)) in zip(feed_dict.values(), self.input_metadata.values())]
            start = time.time()
            outputs = self.model(*inputs)
            end = time.time()

        out_dict = OrderedDict()
        for name, output in zip(self.output_names, outputs):
            out_dict[name] = output.cpu().numpy()
        return out_dict, end - start


    def get_input_metadata(self):
        return self.input_metadata
