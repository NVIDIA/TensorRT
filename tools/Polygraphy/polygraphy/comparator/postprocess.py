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
from polygraphy import mod, util

np = mod.lazy_import("numpy")


@mod.export()
class PostprocessFunc(object):
    """
    Provides functions that can apply post-processing to `IterationResult` s.
    """

    @staticmethod
    # This function returns a top_k function that can be used as a postprocess_func.
    def topk_func(k=10, axis=-1):
        """
        Creates a function that applies a Top-K operation to a IterationResult.
        Top-K will return the indices of the k largest values in the array.

        Args:
            k (Union[int, Dict[str, int]]):
                    The number of indices to keep.
                    If this exceeds the axis length, it will be clamped.
                    This can be specified on a per-output basis by provided a dictionary. In that case,
                    use an empty string ("") as the key to specify default top-k value for outputs not explicitly listed.
                    Defaults to 10.
            axis (int):
                    The axis along which to apply the topk.
                    Defaults to -1.

        Returns:
            Callable(IterationResult) -> IterationResult: The top-k function.
        """
        # Top-K implementation.
        def topk(iter_result):
            for name, output in iter_result.items():
                k_val = util.value_or_from_dict(k, name)
                if k_val:
                    indices = np.argsort(-output, axis=axis, kind="stable")
                    axis_len = indices.shape[axis]
                    iter_result[name] = np.take(indices, np.arange(0, min(k_val, axis_len)), axis=axis)
            return iter_result

        return topk
