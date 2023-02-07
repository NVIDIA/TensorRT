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
from polygraphy import constants, mod, util
from polygraphy.backend.trt import util as trt_util
from polygraphy.common.interface import TypedDict
from polygraphy.logger import G_LOGGER, LogMode


@mod.export()
class ShapeTuple:
    """
    Represents a set of shapes for a single binding in a profile.
    """

    def __init__(self, min, opt, max):
        """
        Args:
            min (Tuple[int]): The minimum shape that the profile will support.
            opt (Tuple[int]): The shape for which TensorRT will optimize the engine.
            max (Tuple[int]): The maximum shape that the profile will support.
        """
        self.min = min
        self.opt = opt
        self.max = max

    def __str__(self):
        return f"(min={self.min}, opt={self.opt}, max={self.max})"

    def __repr__(self):
        return type(self).__name__ + self.__str__()

    def __iter__(self):
        yield from [self.min, self.opt, self.max]


@mod.export()
class Profile(TypedDict(lambda: str, lambda: ShapeTuple)):
    """
    An ordered dictionary that represents a single optimization profile that
    can be used to build an engine.

    More specifically, it is an ``OrderedDict[str, ShapeTuple]`` which maps binding
    names to a set of min/opt/max shapes.
    """

    def add(self, name, min, opt, max):
        """
        A convenience function to add shapes for a single binding.

        Args:
            name (str): The name of the binding.
            min (Tuple[int]): The minimum shape that the profile will support.
            opt (Tuple[int]): The shape for which TensorRT will optimize the engine.
            max (Tuple[int]): The maximum shape that the profile will support.

        Returns:
            Profile:
                self, which allows this function to be easily chained to add multiple bindings,
                e.g., Profile().add(...).add(...)
        """
        self[name] = ShapeTuple(min, opt, max)
        return self

    def __getitem__(self, key):
        """
        Retrieves the shapes registered for a given input name.

        Returns:
            ShapeTuple:
                    A named tuple including ``min``, ``opt``, and ``max`` members for the shapes
                    corresponding to the input.
        """
        if key not in self:
            G_LOGGER.critical(f"Binding: {key} does not have shapes set in this profile")
        return super().__getitem__(key)

    def fill_defaults(self, network, default_shape_value=None):
        """
        Fill this profile with sane default values for any bindings whose
        shapes have not been set explicitly.

        Args:
            network (trt.INetworkDefinition):
                    The TensorRT network this profile is meant for.
                    This will be used to determine model inputs and their shapes.
            default_shape_value (int):
                    The value to use to override dynamic dimensions.

        Returns:
            Profile: Self
        """
        default_shape_value = util.default(default_shape_value, constants.DEFAULT_SHAPE_VALUE)

        for idx in range(network.num_inputs):
            inp = network.get_input(idx)

            if inp.name in self:
                continue

            with G_LOGGER.verbosity(G_LOGGER.CRITICAL):  # WAR for spam from TRT
                is_shape_tensor = inp.is_shape_tensor
            if is_shape_tensor:
                rank = inp.shape[0] if len(inp.shape) > 0 else 1
                shape = (default_shape_value,) * rank
                G_LOGGER.warning(
                    f"{trt_util.str_from_tensor(inp, is_shape_tensor)} | No values provided; "
                    f"Will use input values: {shape} for min/opt/max in profile.\n",
                    mode=LogMode.ONCE,
                )
                G_LOGGER.warning(
                    "This will cause the shape-tensor to have static values. If this is incorrect, please "
                    "set the range of values for this input shape-tensor.",
                    mode=LogMode.ONCE,
                )
            else:
                shape = util.override_dynamic_shape(inp.shape, default_shape_value)
                if shape != inp.shape:
                    G_LOGGER.warning(
                        f"{trt_util.str_from_tensor(inp, is_shape_tensor)} | No shapes provided; Will use shape: {shape} for min/opt/max in profile.\n",
                        mode=LogMode.ONCE,
                    )
                    G_LOGGER.warning(
                        "This will cause the tensor to have a static shape. If this is incorrect, please "
                        "set the range of shapes for this input tensor.",
                        mode=LogMode.ONCE,
                    )

            self.add(inp.name, shape, shape, shape)
        return self

    def to_trt(self, builder, network):
        """
        Creates a TensorRT IOptimizationProfile based on the values set in this Profile.

        Args:
            builder (trt.Builder):
                    A TensorRT builder. This will be used to construct the IOptimizationProfile.
            network (trt.INetworkDefinition):
                    The TensorRT network the profile applies to.

        Returns:
            trt.IOptimizationProfile: A TensorRT optimization profile.
        """
        trt_profile = builder.create_optimization_profile()
        unused_keys = set(self.keys())
        available_inputs = set()
        for idx in range(network.num_inputs):
            inp = network.get_input(idx)
            if inp.name in unused_keys:
                unused_keys.remove(inp.name)
            available_inputs.add(inp.name)

            with G_LOGGER.verbosity():  # WAR for spam from TRT
                is_shape_tensor = inp.is_shape_tensor

            if is_shape_tensor:
                if inp.name in self:
                    shapes = self[inp.name]
                    trt_profile.set_shape_input(inp.name, shapes.min, shapes.opt, shapes.max)
                    G_LOGGER.verbose(
                        f"{trt_util.str_from_tensor(inp, is_shape_tensor)} | Setting input shape-tensor value range to: {shapes}"
                    )
                else:
                    G_LOGGER.warning(
                        f"{trt_util.str_from_tensor(inp, is_shape_tensor)} | No values provided. Assuming this is not a dynamic shape-tensor.",
                        mode=LogMode.ONCE,
                    )
            else:
                shapes = self[inp.name]
                trt_profile.set_shape(inp.name, shapes.min, shapes.opt, shapes.max)
                G_LOGGER.verbose(
                    f"{trt_util.str_from_tensor(inp, is_shape_tensor)} | Setting input tensor shapes to: {shapes}"
                )

        if unused_keys:
            G_LOGGER.critical(
                f"Invalid inputs were provided to the optimization profile: {unused_keys}\n"
                f"Note: Inputs available in the TensorRT network are: {available_inputs}"
            )

        return trt_util.check_profile(trt_profile)

    def __repr__(self):
        ret = "Profile()"
        for name, (min, opt, max) in self.items():
            ret += f".add('{name}', min={min}, opt={opt}, max={max})"
        return ret

    def __str__(self):
        elems = []
        for name, (min, opt, max) in self.items():
            elems.append(f"{name} [min={min}, opt={opt}, max={max}]")

        sep = ",\n "
        return "{" + sep.join(elems) + "}"
