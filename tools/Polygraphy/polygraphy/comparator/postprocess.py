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
from polygraphy import mod, util

np = mod.lazy_import("numpy")


@mod.export()
class BasePostprocessFunc:
    """
    Base class for post-processing functors, which transform an ``IterationResult`` before it is
    compared. Subclasses implement ``__call__(IterationResult) -> IterationResult`` and are used as
    the ``postprocess_func`` argument to ``Comparator.postprocess``.
    """

    def __call__(self, iter_result):
        raise NotImplementedError()


@mod.export()
class TopKPostprocessFunc(BasePostprocessFunc):
    """
    A post-processing functor that replaces each output with the indices of its K largest values
    (a Top-K operation), optionally along a specified axis. Used as the ``postprocess_func``
    argument to ``Comparator.postprocess``.
    """

    def __init__(self, k=None):
        """
        Args:
            k (Union[int, Tuple[int, int], Dict[str, int], Dict[str, Tuple[int, int]]]):
                    The number of indices to keep and optionally the axis on which to operate.
                    For example, a value of ``(5, 0)`` would keep the top 5 indices along axis 0.

                    If this exceeds the axis length, it will be clamped.
                    This can be specified on a per-output basis by providing a dictionary. In that case,
                    use an empty string ("") as the key to specify default top-k value for outputs not explicitly listed.
                    If no default is present, unspecified outputs will not be modified.
                    Defaults to 10.
        """
        self.k = util.default(k, 10)

    def __call__(self, iter_result):
        """
        Args:
            iter_result (IterationResult): The iteration result to post-process.

        Returns:
            IterationResult: The same ``IterationResult``, modified in place.
        """
        for name, output in iter_result.items():
            k_val = util.value_or_from_dict(self.k, name)
            if k_val:
                axis = -1
                if util.is_sequence(k_val):
                    k_val, axis = k_val
                iter_result[name] = util.array.topk(output, k_val, axis)[1]
        return iter_result


@mod.export()
class PostprocessFunc:
    """
    Provides functions that can apply post-processing to `IterationResult` s.
    """

    @staticmethod
    @mod.deprecate(
        remove_in="0.60.0",
        use_instead="TopKPostprocessFunc",
        name="PostprocessFunc.top_k",
    )
    def top_k(k=None):
        """
        Creates a ``TopKPostprocessFunc``. See ``TopKPostprocessFunc`` for a description of the
        arguments.
        """
        return TopKPostprocessFunc(k=k)
