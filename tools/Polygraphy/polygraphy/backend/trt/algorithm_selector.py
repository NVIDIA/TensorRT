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

from polygraphy import func, mod, util
from polygraphy.backend.trt import util as trt_util
from polygraphy.common.interface import TypedDict
from polygraphy.json import Decoder, Encoder, add_json_methods
from polygraphy.logger import G_LOGGER, LogMode

trt = mod.lazy_import("tensorrt")


##
## Data Structures
##

# NOTE: Modifying the structure of the data classes below will break backwards compatiblity


@mod.export()
class Algorithm(object):
    """
    Represents a TensorRT algorithm variant, which can be uniquely represented
    by an implementation ID and tactic ID.
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
        """

        def unpack_io_info(io_info):
            return (io_info.tensor_format, io_info.dtype)

        implementation = algorithm.algorithm_variant.implementation
        tactic = algorithm.algorithm_variant.tactic
        inputs = tuple(unpack_io_info(algorithm.get_algorithm_io_info(i)) for i in range(context.num_inputs))
        outputs = tuple(
            unpack_io_info(algorithm.get_algorithm_io_info(i))
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
            inputs (List[Tuple[trt.TensorFormat, trt.DataType]]):
                    A list of tuples containg a TensorRT tensor format and data type for each input.
            outputs (List[Tuple[trt.TensorFormat, trt.DataType]]):
                    A list of tuples containg a TensorRT tensor format and data type for each output.
        """

        def validate_meta(meta):
            for (fmt, dtype) in meta:
                if not isinstance(fmt, trt.TensorFormat):
                    G_LOGGER.critical(
                        "'format' must be an instance of trt.TensorFormat, but is: {:}.\n"
                        "Note: Provided input/output metadata was: {:}".format(fmt, meta)
                    )
                if not isinstance(dtype, trt.DataType):
                    G_LOGGER.critical(
                        "'dtype' must be an instance of trt.DataType, but is: {:}.\n"
                        "Note: Provided input/output metadata was: {:}".format(dtype, meta)
                    )
            return meta

        self.implementation = implementation
        self.tactic = tactic
        # Use tuples here so the class is hashable.
        self.inputs = tuple(validate_meta(inputs))
        self.outputs = tuple(validate_meta(outputs))

    def __str__(self):
        def io_str(io):
            return tuple((str(tensor_format), str(dtype)) for tensor_format, dtype in io)

        return "(Implementation: {:}, Tactic: {:}) | Inputs: {:} | Outputs: {:}".format(
            self.implementation, self.tactic, io_str(self.inputs), io_str(self.outputs)
        )

    def __eq__(self, other):
        tactic_matches = self.implementation == other.implementation and self.tactic == other.tactic
        io_matches = self.inputs == other.inputs and self.outputs == other.outputs
        return tactic_matches and io_matches

    def __hash__(self):
        return hash((self.implementation, self.tactic, self.inputs, self.outputs))


@Encoder.register(Algorithm)
def encode(algo):
    def encode_algo_io(io_list):
        encoded = []
        for fmt, dtype in io_list:
            encoded.append((str(fmt), str(dtype)))
        return encoded

    return {
        "implementation": algo.implementation,
        "tactic": algo.tactic,
        "inputs": encode_algo_io(algo.inputs),
        "outputs": encode_algo_io(algo.outputs),
    }


@Decoder.register(Algorithm)
def decode(dct):
    def decode_algo_io(io_list):
        decoded = []
        for fmt, dtype in io_list:
            decoded.append((util.getattr_nested(trt, fmt), util.getattr_nested(trt, dtype)))
        return decoded

    return Algorithm(
        implementation=dct["implementation"],
        tactic=dct["tactic"],
        inputs=decode_algo_io(dct["inputs"]),
        outputs=decode_algo_io(dct["outputs"]),
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
        if not isinstance(algorithm, Algorithm):
            G_LOGGER.critical("Tactic replay data expects Algorithm instances, not: {:}".format(algorithm))

        self.dct[name] = algorithm
        return self

    def __str__(self):
        return "\n".join(["Layer: {:}\n\tAlgorithm: {:}".format(name, algorithm) for (name, algorithm) in self.items()])


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
    ALGO_SELECTOR_ENABLED = False
    if mod.version(trt.__version__) >= mod.version("8.0"):
        ALGO_SELECTOR_ENABLED = True
        IAlgorithmSelector = trt.IAlgorithmSelector
    else:
        IAlgorithmSelector = object

    class BaseSelector(IAlgorithmSelector):
        def __init__(self, data):
            if not ALGO_SELECTOR_ENABLED:
                trt_util.fail_unavailable("Algorithm selector")

            # Must explicitly initialize parent for any trampoline class! Will mysteriously segfault without this.
            IAlgorithmSelector.__init__(self)

            self.path = None
            self.data = TacticReplayData()
            if isinstance(data, TacticReplayData):
                self.data = data
            else:
                self.path = data

        def select_algorithms(self, context, choices):
            return list(range(len(choices)))

    return BaseSelector


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
            for (context, choice) in zip(contexts, choices):
                self.data.add(context.name, Algorithm.from_trt(context, choice))

            if self.path is not None:
                self.data.save(self.path)

    return TacticRecorderClass()


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
                    "Layer: {:} was not found in the tactic replay. Falling back to default tactics.".format(
                        context.name
                    )
                )
                G_LOGGER.warning(
                    "Has the network changed since the tactic replay file was generated?\n"
                    "Note: Layers in the tactic replay are:\n\t{:}".format("\n\t".join(self.data.keys())),
                    mode=LogMode.ONCE,
                )
                return default_choices

            # Need to find the index of the tactic we want.
            to_select = self.data[context.name]
            tactic_choices = [Algorithm.from_trt(context, algo) for algo in choices]

            if to_select not in tactic_choices:
                G_LOGGER.critical(
                    "Layer: {:} | Tactic in replay was not provided by TensorRT as a choice for this layer.\n"
                    "Has the network or builder configuration changed since the replay file was generated?\n"
                    "Note: Tactic in replay was:\n\t{:}\nProvided choices were:\n\t{:}".format(
                        context.name, to_select, "\n\t".join(map(str, tactic_choices))
                    )
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
            for (context, choice) in zip(contexts, choices):
                if context.name in self.data:
                    to_select = self.data[context.name]
                    selected = Algorithm.from_trt(context, choice)
                    if to_select != selected:
                        G_LOGGER.critical(
                            "Layer: {:} | TensorRT selected a tactic different than the one specified in the tactic replay.\n"
                            "Note: Tactic in replay was:\n\t{:}, but TensorRT selected:\n\t{:}".format(
                                context.name, to_select, selected
                            )
                        )

    return TacticReplayerClass()
