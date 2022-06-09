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
from polygraphy import mod, util

np = mod.lazy_import("numpy")


@mod.export()
class PostprocessFunc:
    """
    Provides functions that can apply post-processing to `IterationResult` s.
    """

    @staticmethod
    def topk_func(*args, **kwargs):
        mod.warn_deprecated("topk_func", remove_in="0.40.0", use_instead="top_k")
        return PostprocessFunc.top_k(*args, **kwargs)

    @staticmethod
    # This function returns a top_k function that can be used as a postprocess_func.
    def top_k(k=None, axis=None):
        """
        Creates a function that applies a Top-K operation to a IterationResult.
        Top-K will return the indices of the k largest values in the array.

        Args:
            k (Union[int, Tuple[int, int], Dict[str, int], Dict[str, Tuple[int, int]]]):
                    The number of indices to keep and optionally the axis on which to operate.
                    If this exceeds the axis length, it will be clamped.
                    This can be specified on a per-output basis by providing a dictionary. In that case,
                    use an empty string ("") as the key to specify default top-k value for outputs not explicitly listed.
                    If no default is present, unspecified outputs will not be modified.
                    Defaults to 10.
            axis (int):
                    [Deprecated - specify axis in k parameter instead]
                    The axis along which to apply the Top-K.
                    Defaults to -1.

        Returns:
            Callable(IterationResult) -> IterationResult: The top-k function.
        """
        k = util.default(k, 10)
        if axis is not None:
            mod.warn_deprecated("axis", "the k parameter", remove_in="0.40.0")
        axis = util.default(axis, -1)

        # Top-K implementation.
        def top_k_impl(iter_result):
            for name, output in iter_result.items():
                k_val = util.value_or_from_dict(k, name)
                if k_val:
                    nonlocal axis
                    if util.is_sequence(k_val):
                        k_val, axis = k_val

                    indices = np.argsort(-output, axis=axis, kind="stable")
                    axis_len = indices.shape[axis]
                    iter_result[name] = np.take(indices, np.arange(0, min(k_val, axis_len)), axis=axis)
            return iter_result

        return top_k_impl
