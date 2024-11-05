#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import builtins
import tensorrt as trt
from typing import List, Iterable
import copy

from ._utils import _str_to_data_type
from ._export import public_api


# "onesided" means either type or format combinations. After combinations for each are separately generated, we will combine them later.
# e.g. io_variants = ["FP32|FP16", "FP32|FP16", "FP32*FP16"] for a plugin with 3 I/Os. i.e. I/O indices 0 and 1 are dependently either FP32/FP16 and index 2 is independently FP32/FP16.
# There will be 2 * 2 = 4 combinations here: ["FP32", "FP32", "FP32"], ["FP16", "FP16", "FP32"], ["FP32", "FP32", "FP16"], ["FP16", "FP16", "FP16"]
def _gen_onesided_combinations(io_variants):

    # Algorithm:
    # (1) Ignore independent variants and count the (max) number of dependent variants `mx_poly`
    # (2) Compile initial list of #`mx_poly` combinations using the first option (option 0) for any independent variants
    # (3) For each independent variant IO index, add combinations with that index replaced by option 1, 2, ...

    combinations = []
    mx_poly = 0  # This is the number of dependent variants

    for io_variant in io_variants:
        io_variant_list = io_variant.split("|")

        if len(io_variant_list) > 1:
            if "*" in io_variant:
                raise ValueError(
                    f"Type/Format '{io_variant}' contains both '|' and '*'"
                )
            if mx_poly > 1:
                if mx_poly != len(io_variant_list):
                    raise ValueError(
                        f"Type/Format combinations {io_variants} contain illegal dependent lengths"
                    )

        mx_poly = builtins.max(mx_poly, len(io_variant_list))

    for _ in range(mx_poly):
        combinations.append([None] * len(io_variants))

    for j, io_variant in enumerate(io_variants):
        io_variant_list = io_variant.split("|")

        if len(io_variant_list) == 1:
            if "*" in io_variant:
                io_variant_list = io_variant.split("*")
            for i in range(len(combinations)):
                combinations[i][j] = io_variant_list[0]
        else:
            for k in range(len(io_variant_list)):
                combinations[k][j] = io_variant_list[k]

    for j, io_variant in enumerate(io_variants):
        new_combs = []
        if "*" in io_variant:
            io_variant_list = io_variant.split("*")
            for k in range(1, len(io_variant_list)):
                for c in combinations:
                    new_c = copy.deepcopy(c)
                    new_c[j] = io_variant_list[k]
                    new_combs.append(new_c)
            combinations.extend(new_combs)

    return combinations


class _TypeFormatCombination:
    def __init__(self, num=0):
        self.types = [None] * num
        self.layouts = [None] * num
        self.tactics = []

    def set_types(self, types):
        self.types = types

    def set_layouts(self, layouts=None):
        if isinstance(layouts, List):
            self.layouts = layouts
        else:
            self.layouts = [layouts] * len(self.types)

    def __hash__(self):
        return hash((tuple(self.types), tuple(self.layouts)))

    def __eq__(self, other):
        return (
            isinstance(other, _TypeFormatCombination)
            and self.types == other.types
            and self.layouts == other.layouts
        )

    def __str__(self) -> str:
        return "{" + str(self.types) + ", " + str(self.layouts) + "}"


@public_api()
class AutoTuneCombination:
    def __init__(
        self, io_types: str = None, layouts: str = None, tactics: Iterable[int] = None
    ):
        """
        Construct a set of supported type/format combinations of a plugin's I/O.

        Any custom *tactic* s per each such type/format combination can also be advertised. A tactic is simply another way to
        calculate the output of a plugin for the same type/format combination of the I/O (e.g. if there are multiple kernels available).

        Args:
            io_types (str, optional): A string representation of a type combination.

                Valid format is "type0,type1,...,type#io" where 'type' is of the form "TYPE0[sep]TYPE1[sep]...".

                TYPE is a valid string representation of a `trt.DataType`. These include "FP32" for trt.float32, "FP16" for trt.float16. The string representation of other data types is the same as their name in the trt.DataType enum.


                [sep] is a valid separator, which is either '|' or '*'. Only one of these separators can appear in a given `io_types`.

                (1). '|' indicates a dependent combination: the dependence of the type of one I/O to another I/O. e.g. "FP32|FP16,FP32|FP16" indicates the IO can only be both FP32 or both FP16.

                (2). '*' indicates an independent combination. e.g. "FP32*FP16,FP32|FP16,FP32|FP16" indicates that the first input is independently either FP32 or FP16 regardless of the rest of the IO.

            layouts (str, optional): A string representation of a format combination.

                Valid format is "format0,format1,...,format#io" where 'format' is of the form "FORMAT0[sep]FORMAT1[sep]...".

                FORMAT is a valid string representation of a `trt.TensorFormat`. These are string versions for the enum values of `trt.TensorFormat`. e.g. "LINEAR" for `trt.TensorFormat.LINEAR`.

                [sep] is a valid separator, which is either '|' or '*'. The rules are the same as for `io_types`.

            tactics (Iterable[int], optional): Custom tactics for this type/format combination. Each custom tactic must be a positive integer. Defaults to default tactic (0).

        .. code-block:: python
            :linenos:
            :caption: For a plugin with 3 I/Os, I/O indices 0 and 1 are dependently either FP32/FP16 and index 2 is independently FP32/FP16.

            @trtp.autotune("my::plugin")
            def autotune(inp0: trtp.TensorDesc, inp1: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
                # The following would result in the following type combinations:
                # [FP32, FP32, FP32], [FP16, FP16, FP32], [FP32, FP32, FP16], [FP16, FP16, FP16]
                return [trtp.AutoTuneCombination("FP32|FP16, FP32|FP16, FP32|FP16", "LINEAR", [1, 2])]

        .. code-block:: python
            :linenos:
            :caption: For a plugin with 2 I/Os, the input/output supports either LINEAR or HWC format for FP32 and LINEAR format for FP16.

            @trtp.autotune("my::plugin")
            def autotune(inp0: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
                # Even though (FP16, HWC) is not a valid combination (see next example), TRT should intelligently reject those
                # and pass the following combinations to the impl function:
                # [{FP32, FP32}, {LINEAR, LINEAR}], [{FP32, FP32}, {HWC, LINEAR}], [{FP16, FP32}, {LINEAR, LINEAR}]
                return [trtp.AutoTuneCombination("FP32*FP16, FP32", "LINEAR*HWC, LINEAR", [1, 2])]

        .. code-block:: python
            :linenos:
            :caption: For a plugin with 2 I/Os, the input/output supports either LINEAR or HWC format for FP32 and LINEAR format for FP16 (second method).

            @trtp.autotune("my::plugin")
            def autotune(inp0: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
                # We can use two AutoTuneCombination objects to avoid communicating illegal combinations
                return [trtp.AutoTuneCombination("FP32*FP16, FP32", "LINEAR, LINEAR", [1, 2]), trtp.AutoTuneCombination("FP32, FP32", "HWC, LINEAR", [1, 2])]
        """

        if io_types is not None:
            self.io_types = [s.strip() for s in io_types.split(",")]
            if layouts is None:
                layouts = "LINEAR"
            self.layouts = [s.strip() for s in layouts.split(",")]

            if len(self.layouts) > 1:
                assert len(self.io_types) == len(self.layouts)

            if len(self.io_types) > len(self.layouts):
                assert len(self.layouts) == 1
                self.layouts = [self.layouts[0]] * len(self.io_types)
        else:
            self.io_types = []
            self.layouts = []

        self.combinations = []
        self._tactics = tactics

    def pos(self, pos: Iterable[int], io_types: str, layouts: str = "LINEAR") -> None:
        """
        Specify I/O types and formats for a specified set of I/O indices.

        Args:
            pos (Iterable[int]): I/O indices. Input indices are [0, 1, ..., #inputs - 1] and output indices are [#inputs, #inputs + 1, ..., #inputs + #outputs - 1].
            io_types (str): Data types for these I/O indices.
            layouts (str, optional): Tensor format(s) for these I/O indices. Defaults to "LINEAR".
        Raises:
            ValueError: If types or layouts for any of these I/O indices is already specified.

        .. code-block:: python
            :linenos:
            :caption: For a plugin with 3 I/Os, I/O indices 0 and 1 are dependently either FP32/FP16 and index 2 is independently FP32/FP16.

            @trtp.autotune("my::plugin")
            def autotune(inp0: trtp.TensorDesc, inp1: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
                c = trtp.AutoTuneCombination()
                c.pos([0, 1], "FP32|FP16", "LINEAR")
                c.pos(2, "FP32*FP16") # Omitting format is the same as declaring it to be LINEAR.
                c.tactics([1, 2])
                return [c]
        """
        if max(pos) >= len(self.io_types):
            self.io_types.extend([None] * (max(pos) + 1 - len(self.io_types)))
            self.layouts.extend([None] * (max(pos) + 1 - len(self.layouts)))
            assert len(self.io_types) == len(self.layouts)

        for p in pos:
            if self.io_types[p] is not None:
                raise ValueError(f"Type(s) for position {p} already specified")
            if self.layouts[p] is not None:
                raise ValueError(f"Layout(s) for position {p} already specified")
            self.io_types[p] = io_types
            self.layouts[p] = layouts

    def tactics(self, tactics: Iterable[int]) -> None:
        """
        Specify custom tactics for this type/format combination

        Args:
            tactics (Iterable[int]): Custom tactics. These must be positive integers.
        """
        self._tactics = tactics

    def _generate_combinations(self):

        self.combinations = []

        type_combinations = _gen_onesided_combinations(self.io_types)
        layout_combinations = _gen_onesided_combinations(self.layouts)

        for t in type_combinations:
            for l in layout_combinations:
                c = _TypeFormatCombination(len(self.io_types))
                c.types = [_str_to_data_type(tt) for tt in t]
                c.layouts = [getattr(trt.TensorFormat, ff) for ff in l]
                c.tactics = self._tactics
                self.combinations.append(c)

    def _get_combinations(self):
        self._generate_combinations()
        return self.combinations

    def _check(self, pos, type, layout):
        for i in range(len(self.combinations)):
            if (
                self.combinations[i].types[pos] == _str_to_data_type(type)
                and self.combinations[i].layouts[pos] == layout.name
            ):
                return True
        return False
