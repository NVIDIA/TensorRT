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

from polygraphy import func, mod, util, constants
from polygraphy.backend.trt import util as trt_util
from polygraphy.mod.trt_importer import lazy_import_trt
from polygraphy.common.interface import TypedDict
from polygraphy.json import Decoder, Encoder, add_json_methods
from polygraphy.logger import G_LOGGER, LogMode

from typing import Sequence

trt = lazy_import_trt()


##
## Data Structures
##

#
# NOTE: Modifying the structure of the data classes below will break backwards compatiblity
#


def check_is_instance(obj, cls, name):
    if not isinstance(obj, cls):
        G_LOGGER.critical(
            f"'{name}' must be an instance of {cls.__name__}, but is: {obj}."
        )


@mod.export()
class TensorInfo:
    """
    Tracks information about a tensor, such as format and data type.
    """

    @staticmethod
    def from_trt(io_info):
        """
        Creates a Polygraphy ``TensorInfo`` instance from a TensorRT ``IAlgorithmIOInfo``.

        Args:
            io_info (trt.IAlgorithmIOInfo): The algorithm I/O information.

        Returns:
            TensorInfo
        """
        return TensorInfo(
            io_info.dtype,
            tuple(io_info.strides),
            # These fields were added in 8.6
            util.try_getattr(io_info, "vectorized_dim"),
            util.try_getattr(io_info, "components_per_element"),
        )

    def __init__(self, dtype, strides, vectorized_dim, components_per_element):
        """
        Args:
            dtype (trt.DataType): The data type.
            strides (Sequence[int]): The strides.
            vectorized_dim (int): The index of the vectorized dimensions.
            components_per_element (int): The number of components per element.
        """
        check_is_instance(dtype, trt.DataType, "dtype")
        check_is_instance(strides, Sequence, "strides")
        if vectorized_dim is not None:
            check_is_instance(vectorized_dim, int, "vectorized_dim")
        if components_per_element is not None:
            check_is_instance(components_per_element, int, "components_per_element")

        self.dtype = dtype
        self.strides = tuple(strides)
        self.vectorized_dim = vectorized_dim
        self.components_per_element = components_per_element

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return f"TensorInfo({str(self.dtype)}, {self.strides}, {self.vectorized_dim}, {self.components_per_element})"

    def __hash__(self):
        return hash(
            (self.dtype, self.strides, self.vectorized_dim, self.components_per_element)
        )


@Encoder.register(TensorInfo)
def encode(tensor_info):
    return {
        "dtype": str(tensor_info.dtype),
        "strides": tensor_info.strides,
        "vectorized_dim": tensor_info.vectorized_dim,
        "components_per_element": tensor_info.components_per_element,
    }


@Decoder.register(TensorInfo)
def decode(dct):
    return TensorInfo(
        util.getattr_nested(trt, dct["dtype"]),
        dct["strides"],
        dct["vectorized_dim"],
        dct["components_per_element"],
    )


@mod.export()
class Algorithm:
    """
    Represents a TensorRT algorithm variant, which can be uniquely represented
    by an implementation ID, tactic ID, and I/O tensor information.
    """

    @staticmethod
    def from_trt(context, algorithm):
        """
        Creates a Polygraphy ``Algorithm`` instance from a TensorRT
        ``IAlgorithmContext`` and ``IAlgorithm``.

        Args:
            context (trt.IAlgorithmContext):
                    The algorithm context corresponding to the layer.
            algorithm (trt.IAlgorithm):
                    The algorithm variant provided by TensorRT.

        Returns:
            Algorithm
        """

        implementation = algorithm.algorithm_variant.implementation
        tactic = algorithm.algorithm_variant.tactic
        inputs = tuple(
            TensorInfo.from_trt(algorithm.get_algorithm_io_info(i))
            for i in range(context.num_inputs)
        )
        outputs = tuple(
            TensorInfo.from_trt(algorithm.get_algorithm_io_info(i))
            for i in range(context.num_inputs, context.num_inputs + context.num_outputs)
        )
        return Algorithm(implementation, tactic, inputs, outputs)

    def __init__(self, implementation, tactic, inputs, outputs):
        """
        Args:
            implementation (int):
                    The implementation for this Algorithm.
            tactic (int):
                    The tactic for this Algorithm.
            inputs (Sequence[TensorInfo]):
                    A sequence of TensorInfos for each input.
            outputs (Sequence[TensorInfo]):
                    A sequence of TensorInfos for each output.
        """
        self.implementation = implementation
        self.tactic = tactic

        def check_io(lst, name):
            for index, io in enumerate(lst):
                check_is_instance(io, TensorInfo, f"{name}[{index}]")

        check_io(inputs, "inputs")
        check_io(outputs, "outputs")

        # Use tuples here so the class is hashable.
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)

    def __str__(self):
        return f"(Implementation: {self.implementation}, Tactic: {self.tactic}) | Inputs: {self.inputs} | Outputs: {self.outputs}"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash((self.implementation, self.tactic, self.inputs, self.outputs))


@Encoder.register(Algorithm)
def encode(algo):
    return {
        "implementation": algo.implementation,
        "tactic": algo.tactic,
        "inputs": algo.inputs,
        "outputs": algo.outputs,
    }


@Decoder.register(Algorithm)
def decode(dct):
    return Algorithm(
        implementation=dct["implementation"],
        tactic=dct["tactic"],
        inputs=dct["inputs"],
        outputs=dct["outputs"],
    )


@mod.export()
@add_json_methods("tactic replay file")
class TacticReplayData(TypedDict(lambda: str, lambda: Algorithm)):
    """
    Maps layer names to corresponding tactics.
    More specifically, it is an ``OrderedDict[str, Algorithm]``.
    """

    def add(self, name, algorithm):
        """
        Add an entry into the tactic replay data.

        Args:
            name (str): The name of the layer
            algorithm (Algorithm): The algorithm to use for the layer.

        Returns:
            TacticReplayData: self, to allow for method chaining.
        """
        self[name] = algorithm
        return self

    def __str__(self):
        return "\n".join(
            [
                f"Layer: {name}\n{constants.TAB}Algorithm: {algorithm}"
                for (name, algorithm) in self.items()
            ]
        )


@Encoder.register(TacticReplayData)
def encode(replay):
    return {"replay": replay.dct}


@Decoder.register(TacticReplayData)
def decode(dct):
    return TacticReplayData(dct["replay"])


##
## Algorithm Selectors
##


# Everything is encapsulated in functions so that we don't create a dependency on TensorRT
# when objects from this file are imported.
def get_base_selector_type():
    class BaseSelector(trt.IAlgorithmSelector):
        def __init__(self, data):
            # Must explicitly initialize parent for any trampoline class! Will mysteriously segfault without this.
            trt.IAlgorithmSelector.__init__(self)

            self.path = None
            self.data = TacticReplayData()
            if isinstance(data, TacticReplayData):
                self.data = data
            else:
                self.path = data

        def select_algorithms(self, context, choices):
            return list(range(len(choices)))

    return BaseSelector


@mod.deprecate(remove_in="0.50.0", use_instead=None)
@mod.export()
def TacticRecorder(record):
    """
    A TensorRT algorithm selector that can record tactics selected by TensorRT.

    The generated tactic replay file is specific to network and builder configuration.
    Changing either of these may render the tactic replay file unusable.

    Args:
        record (Union[path, file-like, TacticReplayData]):
                A path or file-like object or an empty ``TacticReplayData`` instance.
                Tactics will be recorded and stored here.
    """

    class TacticRecorderClass(get_base_selector_type()):
        def __init__(self):
            super().__init__(record)
            # The function that constructed this instance
            self.make_func = TacticRecorder

        @G_LOGGER.log_exception
        def report_algorithms(self, contexts, choices):
            """
            Records algorithms selected by TensorRT into the provided path or
            ``TacticReplayData`` instance.

            Args:
                contexts (List[trt.IAlgorithmContext]):
                        The list of TensorRT algorithm contexts. Generally, there is one per layer.
                choices (List[trt.IAlgorithm]):
                        A list of selected algorithms for each context.

            Returns:
                None
            """
            for context, choice in zip(contexts, choices):
                self.data.add(context.name, Algorithm.from_trt(context, choice))

            if self.path is not None:
                self.data.save(self.path)

    return TacticRecorderClass()

@mod.deprecate(remove_in="0.50.0", use_instead=None)
@mod.export()
def TacticReplayer(replay):
    """
    A TensorRT algorithm selector that can replay tactics according to a tactic replay file.

    Args:
        replay (Union[path, file-like, TacticReplayData]):
                A path or file-like object containing a JSON-ified ``TacticReplayData`` instance,
                or a ``TacticReplayData`` instance.
    """

    class TacticReplayerClass(get_base_selector_type()):
        def __init__(self):
            super().__init__(replay)

            if self.path is not None:
                self.data = TacticReplayData.load(self.path)

            # The function that constructed this instance
            self.make_func = TacticReplayer

        @G_LOGGER.log_exception
        @func.constantmethod
        def select_algorithms(self, context, choices):
            """
            Selects an algorithm based on ``self.data`` if possible. Otherwise, returns
            default tactics.

            Args:
                context (trt.IAlgorithmContext):
                        The TensorRT algorithm context.
                choices (List[trt.IAlgorithm]):
                        A list of TensorRT algorithm choices.

            Returns:
                List[int]:
                        The indices of selected tactics. If ``self.data`` includes the layer and
                        TensorRT provides a matching tactic, this will always be of length 1.

            Raises:
                PolygraphyException:
                        If a tactic is set for a layer in ``self.data`` but is not provided by
                        TensorRT as a choice for that layer.
            """
            default_choices = super().select_algorithms(context, choices)

            if not self.data:  # No replay data, we are in recording mode.
                return default_choices

            if context.name not in self.data:
                G_LOGGER.warning(
                    f"Layer: {context.name} was not found in the tactic replay. Falling back to default tactics."
                )
                sep = f"\n{constants.TAB}"
                G_LOGGER.warning(
                    "Has the network changed since the tactic replay file was generated?\n"
                    f"Note: Layers in the tactic replay are:{sep}{sep.join(self.data.keys())}",
                    mode=LogMode.ONCE,
                )
                return default_choices

            # Need to find the index of the tactic we want.
            to_select = self.data[context.name]
            tactic_choices = [Algorithm.from_trt(context, algo) for algo in choices]

            if to_select not in tactic_choices:
                sep = f"\n{constants.TAB}"
                G_LOGGER.critical(
                    f"Layer: {context.name} | Tactic in replay was not provided by TensorRT as a choice for this layer.\n"
                    f"Has the network or builder configuration changed since the replay file was generated?\n"
                    f"Note: Tactic in replay was:{sep}{to_select}\nProvided choices were:{sep}{sep.join(map(str, tactic_choices))}"
                )

            return [tactic_choices.index(to_select)]

        @G_LOGGER.log_exception
        @func.constantmethod
        def report_algorithms(self, contexts, choices):
            """
            Checks if the tactics specified in ``self.data`` were selected and raises an exception
            if not.

            Raises:
                PolygraphyException:
                        If a tactic specified in ``self.data`` was not selected for a layer.
            """
            for context, choice in zip(contexts, choices):
                if context.name in self.data:
                    to_select = self.data[context.name]
                    selected = Algorithm.from_trt(context, choice)
                    if to_select != selected:
                        G_LOGGER.critical(
                            f"Layer: {context.name} | TensorRT selected a tactic different than the one specified in the tactic replay."
                            f"\nNote: Tactic in replay was:\n{constants.TAB}{to_select}, but TensorRT selected:\n{constants.TAB}{selected}"
                        )

    return TacticReplayerClass()
