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

import contextlib
import enum
import functools
import io
import json
import os
import re
import sys
import tempfile
from collections import OrderedDict
from typing import Optional, Union

from polygraphy import mod
from polygraphy.comparator import IterationResult
from polygraphy.exception import PolygraphyException
from polygraphy.json import save_json
from polygraphy.logger import G_LOGGER
from polygraphy.tools import util as tools_util
from polygraphy.tools.args import (
    DataLoaderArgs,
    ModelArgs,
    OnnxLoadArgs,
    OnnxrtSessionArgs,
)
from polygraphy.tools.base import Tool

onnx = mod.lazy_import("onnx")
gs = mod.lazy_import("onnx_graphsurgeon>=0.3.21")
onnx_util = mod.lazy_import("polygraphy.backend.onnx.util")
onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
onnxrt_backend = mod.lazy_import("polygraphy.backend.onnxrt")


class Lint(Tool):
    """
    [EXPERIMENTAL] Topologically "lint" an ONNX model to find faulty nodes in the graph.
    All nodes that depend on a faulty node will be marked as faulty and ignored.

    All error messages and warnings are captured in a JSON report.

    The JSON report contains the following fields:
    - 'summary' : summarizes the passing and failing nodes among the ones that are linted.
    (Note: the nodes included are not exhaustive, as some nodes may be skipped due to dependency on a faulty previous node)
    - 'linting_entries': a list of linting entries, each of which contains the following fields:
        - 'level': the severity of the linting entry (error or warning)
        - 'source': The underlying checker that generated the error message (either `onnx.checker` or ONNX Runtime)
        - 'message': The error message. This message is superficially parsed/pruned but may retain formatting of the underlying checker.
        - (optional) 'nodes': A list of nodes that are related to the error message. If this field is not present,
            then the linting entry is a global error/warning that applies to the entire model (like a missing opset import).

    The schema for the json output is:
        {
            'summary': {
                'passing': [<list of nodes that passed ORT inference check>],
                'failing': [<list of nodes that failed ORT inference check>],
                },
            'lint_entries': [
                { 'level': <severity level>, 'source': <source of error>, 'message': <error string>, 'nodes': [<name of failing node>] },
                ...
            ]
        }

    Known Limitations:
    ------------------
    1. BFLOAT16 and FLOAT8  are not currently supported.
    2. Only erroneous nodes that are independent of each other are captured in the JSON report. Downstream nodes that depend on a faulty node are not checked.
    3. Subgraph nested inside nodes are not recursively linted.
    4. Custom Ops are documented as warnings in the JSON Report, but are treated as exceptions by the internal inference checks. Therefore downstream nodes that depend on the custom op are not checked for error or custom ops.
    5. The subtool verifies data-dependent failures either based on user's input data or generating random data for the input tensors. Therefore, the subtool's coverage of subgraphs are completely dependent on the input data and does not guarantee 100% coverage.
    For example, if a subgraph has a conditional branch, the subtool will only check the branch that is taken based on the input data.
    6. Large models (>2GB) require external data to be in same directory as the model file, custom paths to external data are not supported.
    """

    CUSTOM_OP_EXCEPTION_SUBSTRS = [
        "No opset import for domain",
        "is not a registered function/op",
    ]
    ONNX_CHECKER_IGNORE_SUBSTR = "Bad node spec for node"
    INVALID_ONNX_EXCEPTION_SUBSTR = "Error parsing message with type 'onnx.ModelProto'"
    MAXIMUM_PROTOBUF = 2e9  # 2GB

    class ContextManager:
        """
        Keeps track of the linting process, including the current node being linted, cached tensors and their consumers.
        Provides an interface to explicitly perform inference node-by-node, for node-level control of the computational graph.
        """

        def __init__(self, graph: "gs.Graph"):
            """
            Args:
                graph (gs.Graph):
                    The input graphsurgeon graph to be linted.

            Attributes:
                graph (gs.Graph):
                    The input graph reference
                tensor_map (OrderedDict[str, Tensor]):
                    Mapping of tensor names to tensors in the graph.
                cur_node (gs.Node):
                    Initially set to None, represents the current node being processed in the graph.
                num_consumers (OrderedDict[str, int]):
                    Keeps track of the consumers for each tensor cached
                cache (OrderedDict[str, Tensor]):
                    Keeps track of tensor data, used for feeding inference.
            """

            self.graph = graph
            self.tensor_map = OrderedDict()
            self.cur_node = None
            self.num_consumers = OrderedDict()
            self.cache = IterationResult()

        def __enter__(self):
            """
            Enter the context of the linting process.
            """
            self.tensor_map = self.graph.tensors()
            for tensor in self.tensor_map.values():
                if isinstance(tensor, gs.Variable):
                    # Set the number of consumers for each tensor
                    self.num_consumers[tensor.name] = len(tensor.outputs)

            return self

        def nodes(self) -> "gs.Node":
            """
            Get the next node to be linted. Nodes are yielded in topological order.
            """
            for node in self.graph.nodes:
                self.cur_node = node
                G_LOGGER.extra_verbose(
                    f"Linting node: {node.name}: {node.op}({[inp.name for inp in node.inputs]})->{[out.name for out in node.outputs]}"
                )
                yield node

        def make_singleton_graph(self) -> Optional["gs.Graph"]:
            """
            Creates a singleton graph with just the current node, its inputs and outputs.

            This function first checks if all the inputs for the current node are available in the cache. If not, it returns None.
            If all inputs are available, it creates a subgraph with only the current node and its inputs.
            The input metadata of the singleton graph is then updated to be used for inference.

            Returns:
                singleton (gs.Graph):
                    The singleton graph created from the current node, its inputs and outputs.
                    If not all inputs for the current node are available in the cache, the function returns None.

            """
            node = self.cur_node

            inp_names = {
                inp.name for inp in node.inputs if isinstance(inp, gs.Variable)
            }

            if not all(
                [inp in self.cache for inp in inp_names]
            ):  # Need all inputs to be available in the cache
                return None

            singleton = self.graph.copy()
            singleton.nodes = [node]
            singleton.inputs = node.inputs
            singleton.outputs = node.outputs
            singleton.name = node.name

            # Update the input metadata of the singleton graph so that it can be used for inference
            # NOTE: nodes can treat the same tensor as two or more inputs, but a graph should be defined with uniquely named value infos.
            singleton_input_dict = {
                inp.name: inp.to_variable(
                    shape=self.cache[inp.name].shape,
                    dtype=self.cache[inp.name].dtype,
                )
                for inp in singleton.inputs
                if isinstance(inp, gs.Variable)
            }
            singleton.inputs = list(singleton_input_dict.values())

            return singleton

        def update(self, output_dict: Optional[dict]):
            """
            Update the cache and available tensors after linting a node.
            This should be called after the current node has been linted.
            """
            # Now the node has been visited, the node's inputs have leser consumers
            for inp in self.cur_node.inputs:
                if inp.name not in self.cache:
                    G_LOGGER.super_verbose(
                        f"node `{self.cur_node.name}`'s input tensor: `{inp.name}` missing in cache. something wrong with node's ancestors."
                    )
                    # If some inputs for current node are missing,
                    # means that something went wrong with its ancestor nodes.
                    continue
                self.num_consumers[inp.name] -= 1
                if (
                    self.num_consumers[inp.name] == 0
                ):  # All consuming nodes of this tensor have been visited
                    G_LOGGER.super_verbose(f"removing tensor: `{inp.name}` from cache")
                    del self.cache[inp.name]  # Can delete the tensor from the cache

            if not output_dict:
                return

            # Update the cache with the outputs of the current node
            for name in output_dict.keys():
                out = self.tensor_map[name]
                if isinstance(out, gs.Variable):
                    G_LOGGER.super_verbose(f"adding tensor: `{out.name}` to cache")
                    self.cache[out.name] = output_dict[name]
                elif isinstance(
                    out, gs.Constant
                ):  # This theoretically should never happen, as constants are not outputs of nodes
                    G_LOGGER.critical(
                        f"tensor: `{out.name}` is a constant, but is part of the output!"
                    )
                else:
                    G_LOGGER.critical(
                        f"tensor: `{out.name}` is neither a variable nor a constant"
                    )

        def set_graph_inputs(self, feed_dict: dict):
            """
            Initialize the cache with the input feed_dict for source node.
            """
            self.cache = feed_dict

        def feed_dict(self) -> dict:
            """
            Provide a feed_dict for the current node from cache.
            Expects that all inputs of the current node are available in the cache.
            """
            _feed_dict = {}
            for inp in self.cur_node.inputs:
                if inp.name not in self.cache:
                    if isinstance(inp, gs.Variable):
                        G_LOGGER.internal_error(
                            f"tensor: {inp.name} missing in input cache! are you sure current node {self.cur_node.name} is valid?"
                        )  # This should never happen
                    elif isinstance(inp, gs.Constant):
                        G_LOGGER.super_verbose(
                            f"tensor: `{inp.name}` is a constant, not tracked in cache. "
                        )
                        continue

                _feed_dict[inp.name] = self.cache[inp.name]
            return _feed_dict

        def __exit__(self, exc_type, exc_value, traceback):
            """
            Exit the context of the linting process.
            """
            G_LOGGER.ultra_verbose("exiting lint context")
            self.num_consumers = {}
            self.cache = {}
            self.cur_node = None

    class Level(enum.Enum):  # Severity of linting message
        EXCEPTION = "exception"
        WARNING = "warning"
        INFO = "info"

    class Source(enum.Enum):  # Source of the error message
        ONNXRUNTIME = "onnxruntime"
        ONNX_CHECKER = "onnx_checker"
        ONNX_LOADER = "onnx_loader"
        ONNX_GS = "onnx_graphsurgeon"

    class Report:
        """
        Record the Linting report.

        The report is a dictionary with the following structure:
        {
            'summary': {
                'passing': [list of nodes that passed ORT inference check],
                'failing': [list of nodes that failed ORT inference check],
                },
            'lint_entries': [
                { 'level': Lint.Level, 'source': str, 'message': str, 'nodes': [node_name] },
                ...
            ]
        }

        """

        def __init__(self):
            self.lint_entries = []
            self.is_model_valid = True
            self.summary = {
                "passing": set(),
                "failing": set(),
            }

        def add(
            self,
            level: "Lint.Level",
            source: "Lint.Source",
            message: Optional[str] = None,
            node_name: Optional[str] = None,
            op: Optional[str] = None,
            log: bool = True,
        ):
            """
            Adds a lint entry to the report and updates the summary dictionary.

            This method performs two major functions under the hood:

            1. updates the passing and failing nodes in the summary dictionary `self.summary`.
            The `node_name` is added to the `passing` or `failing` list based on the `level` and `message`.
                If `node_name` is not None, the following logic is used to determine if the node is passing or failing:
                    - If `message` is None, the node is marked as passing, irrespecitive of the `level` if that node isn't already
                    in the failing set.
                    - If `message` is not None, and `level` is `Lint.Level.EXCEPTION`, the node is marked as failing,
                    and removed from the passing set if exists.

            2. Parses the lint entry's message using the `_prune` method before adding the entry to the report.
               This helper method attempts to reverse engineer the formatting done in the ORT codebase to make the error message more readable.

            Args:
                level (Lint.Level):
                    The severity level of the lint entry.
                source (Lint.Source):
                    The source of the lint entry.
                message (str, optional):
                    The message associated with the lint entry. If present, it will be parsed by `_parse_ort_error` method before being added to the report. Defaults to None. If not present, the `node_name` associated (if not None) is marked as passing in the summary dictionary.
                node_name (str, optional):
                    The name of the node associated with the lint entry. If present, the node is marked as passing or failing in the summary dictionary based on the `level` and `message`. Defaults to None.
                op (str, optional):
                    The operator of the linted node (if any). Defaults to None.
                log (bool, optional):
                    If True, the lint entry is printed to the console. Defaults to True.
            """
            if message:
                message = self._prune(message)
                if log:
                    severity_from_level = {
                        Lint.Level.EXCEPTION: G_LOGGER.ERROR,
                        Lint.Level.WARNING: G_LOGGER.WARNING,
                        Lint.Level.INFO: G_LOGGER.INFO,
                    }
                    scope = ""
                    if node_name and op:
                        scope = f"Name: {node_name}, Op: {op} | "
                    G_LOGGER.log(
                        f"LINT | {scope}{message}", severity=severity_from_level[level]
                    )
                lint_entry = {
                    "level": level.value,
                    "source": source.value,
                    "message": message,
                }

                if node_name:
                    lint_entry["nodes"] = [node_name]
                    if level == Lint.Level.EXCEPTION:
                        self.summary["failing"].update([node_name])
                        # Remove from passing set if exists
                        self.summary["passing"].discard(node_name)

                    elif node_name not in self.summary["failing"]:
                        # Only add to passing set if not already in failing set
                        self.summary["passing"].update([node_name])

                self.lint_entries.append(lint_entry)

                self.is_model_valid = (
                    level != Lint.Level.EXCEPTION
                ) and self.is_model_valid

            elif node_name not in self.summary["failing"]:
                self.summary["passing"].update([node_name])

        def export(self, path: str):
            """
            Write the report to a json file.
            """
            report = {
                "summary": {k: list(v) for k, v in self.summary.items()},
                "lint_entries": self.lint_entries,
            }
            G_LOGGER.ultra_verbose(f"report:\n {json.dumps(report, indent=4)}")
            if path:
                save_json(report, path, description="linting report")

        def _prune(self, message: str) -> str:
            """
            Prunes the formatting of the error message that is thrown from ONNX Runtime.
            Essentially attempts to reverse engineer the formatting done in the ORT codebase to make the error message more readable.
            Note: Not exhaustive, some error messages may retain formatting.
            """

            def _prune_ONNXRuntimeError_formatting(message):
                """
                Prunes formating: [ONNXRuntimeError] : {code} : {StatusCodeToString(code)} : {msg}
                and returns only {msg}.
                """
                ORT_SUBSTRS_TO_PRUNE = [
                    "This is an invalid model. ",
                    "Error: ",
                    "Failed to load model with error: ",
                    "Exception caught: ",
                    "Exception during loading: ",
                    "\n",
                ]
                parts = message.split(" : ")
                if len(parts) < 4:
                    # The ORT message format is not as expected, so just return the message pruning the prefix
                    return message.split("[ONNXRuntimeError] : ")[1]
                message = "".join(parts[3:]).replace('"', "`")
                for (
                    substr
                ) in (
                    ORT_SUBSTRS_TO_PRUNE
                ):  # remove substrings that are not useful in the error message
                    message = message.replace(substr, "")
                return message

            # some patterns that were observed while testing various types of error messages
            pattern_prune_dict = {
                r"SystemError: .*": lambda x: G_LOGGER.internal_error(
                    "SystemError: " + x.split(" : ")[1]
                ),  # If starts with "SystemError", it is likely due to improper installation of ONNX Runtime.
                r"\[ONNXRuntimeError\] : .*": _prune_ONNXRuntimeError_formatting,  # [ONNXRuntimeError] : {code} : {StatusCodeToString(code)} : {msg}
                r"\x1b(?:\[(?:\d+;){0,2}\d+m)(.*)\x1b\[m": lambda msg: re.sub(
                    r"\x1b(?:\[(?:\d+;){0,2}\d+m)(.*)\x1b\[m", "\\1", msg
                ),  # Remove log coloration characters from https://github.com/microsoft/onnxruntime/blob/b33216be4c02adfbbdeac2fd30ddc55f673eda3d/onnxruntime/core/common/logging/sinks/ostream_sink.cc#L24
                r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \[.*?\]\ ": lambda msg: re.sub(
                    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \[.*?\]\ ", "", msg
                ),  # (e.g: https://github.com/microsoft/onnxruntime/blob/24566058b3e5bb9e511513977cee6e7c553fd5c2/onnxruntime/core/graph/graph.cc#L3545-L3546)
                r"Node \(.*\) Op \(.*\) \[ShapeInferenceError\]": lambda msg: re.sub(
                    r"Node \(.*\) Op \(.*\) \[ShapeInferenceError\]", "", msg
                ),  # Eg: "Node {name}: Op {op}: [ShapeInferenceError] {msg}"
                r".*/.*\.cc:\d+ onnxruntime::.*\(.*\)": lambda msg: re.sub(
                    r".*/.*\.cc:\d+ onnxruntime::.*\(.*\)", "", msg
                ),  # Eg: {path/to/file}.cc:{line} onnxruntime::{func}({args}) {msg}
                r"In Node,\ .*,\ Error\ ": lambda msg: re.sub(
                    r"In Node,\ .*,\ Error\ ", "", msg
                ),  # Eg: "In Node, {node details}, Error {msg}"
                r".*Status Message:\ ": lambda msg: re.sub(
                    r".*Status Message:\ ", "", msg
                ),  # Eg: "Non-zero status code returned while running {op} node. Name:'{name}' Status Message: {msg}"
            }

            # catches a pattern, and the function substitutes the pattern with ""
            for pattern, func in pattern_prune_dict.items():
                # NOTE: patterns are incrementally matched and modified
                # ONNRuntime's default error message format is first stripped off (if it exists),
                # and then the remaining message is parsed for other patterns
                if re.match(pattern, message):
                    message = func(message)
            return message.replace("\n", "")

    def __init__(self):
        super().__init__("lint")

    def get_subscriptions_impl(self):
        return [
            ModelArgs(model_opt_required=True, required_model_type="onnx"),
            OnnxLoadArgs(outputs_opt_prefix=False, allow_shape_inference=False),
            DataLoaderArgs(),
            OnnxrtSessionArgs(),
        ]

    def show_start_end_logging_impl(self, args):
        return True

    def add_parser_args_impl(self, parser):
        parser.add_argument(
            "-o",
            "--output",
            help="Path to save json report.",
        )

    def prepare_feed_dict(self, onnx_model: "onnx.ModelProto") -> OrderedDict:
        """
        Prepare the feed_dict for the source node of the graph using `DataLoaderArgs`.
        This converts data to Polygraphy's internal DataType format.
        """
        input_metadata = onnx_util.get_input_metadata(onnx_model.graph)
        data_loader = self.arg_groups[DataLoaderArgs].get_data_loader(input_metadata)
        return next(iter(data_loader))

    def load_helper(self, args) -> Optional["onnx.ModelProto"]:
        """
        Loads the ONNX model using `OnnxLoadArgs` and returns the ONNX model.
        If the model is invalid, returns None.
        """
        try:
            onnx_model = self.arg_groups[OnnxLoadArgs].load_onnx()
        except Exception as err:  # pylint: disable=broad-except
            # intentionally catching broad exception to avoid introducing
            # `google.protobuf.message.DecodeError` dependency
            if Lint.INVALID_ONNX_EXCEPTION_SUBSTR in str(err):
                self.report.add(Lint.Level.EXCEPTION, Lint.Source.ONNX_LOADER, str(err))
                self.report.export(args.output)
                G_LOGGER.error(f"Invalid ONNX model given: {err}")
            else:
                # some unkown error
                G_LOGGER.critical(f"Unhandled error: {err}")
            return None

        # if the model is empty with no onnx metadata
        if onnx_model.ByteSize() == 0:
            self.report.add(
                Lint.Level.EXCEPTION,
                Lint.Source.ONNX_LOADER,
                "Empty ONNX model given",
            )
            self.report.export(args.output)
            return None

        return onnx_model

    def run_impl(self, args):
        def _handle_empty_names(graph: "gs.Graph"):
            """
            Handle nodes with empty names in the graph
            by renaming them to "polygraphy_unnamed_node_<id>"
            where <id> is the topological sort order of the node.

            If the above name already exists, then the node is renamed to
            "polygraphy_unnamed_node_<id>_<uid>" where <uid> is a unique id.
            """
            uid = 0
            with graph.node_ids():

                def _generate_unique_name(node_id):
                    nonlocal uid
                    name = f"polygraphy_unnamed_node_{node_id}"
                    names = {node.name for node in graph.nodes}
                    while name in names:  # guarantee unique name
                        name = f"polygraphy_unnamed_node_{node_id}_{uid}"
                        uid += 1
                    G_LOGGER.verbose(
                        f"Node with topological id: {node_id} has empty name. Renaming to: {name}"
                    )
                    return name

                for node in graph.nodes:
                    node.name = node.name or _generate_unique_name(node.id)

        def _duplicate_node_name_check(graph: "gs.Graph"):
            """
            Duplicate names that are non-empty violate ONNX Naming rules.
            This is not caught by ONNX Checker for some reason, hence we need to catch it here.
            """
            name_tracker = {}
            with graph.node_ids():
                for node in graph.nodes:
                    name_tracker.setdefault(node.name, []).append(node.id)

                for name, ids in name_tracker.items():
                    if len(ids) > 1:
                        self.report.add(
                            Lint.Level.EXCEPTION,
                            Lint.Source.ONNX_GS,
                            f"Duplicate node name: '{name}' for nodes with topological IDs: {ids} found.",
                            node_name=name,
                        )

        def _onnx_spec_check(onnx_model: "onnx.ModelProto") -> bool:
            """
            Check the ONNX model for specification errors using `onnx.checker.check_model`.

            Args:
                onnx_model (ModelProto): The ONNX Model to check
            Returns:
                bool: True if the ONNX Checker passes, False otherwise.

            Only Graph-level ONNX metadata as well as inputs are checked for correctness here.
            Node-level specification errors are caught and ignored, as they will be caught by the linting process incrementally.

            Performing this for correct ONNX specifications in the graph-level is important as a pre-linting step
            before checking correctness of each node seperately.
            For example, if an ONNX Graph with duplicated inputs is passed, this will not be caught when linting at the node-level.

            The checks performed are:
            1. ModelProto-level:
                check if opset_imports is non-empty in onnx model
                check for duplicate keys in metadata_props
                check if IR is set in onnx model
            2. GraphProto-level:
                check non-empty name for graphs, tensor-initializers, sparse-tensor-initializers.
                Check nodes are topologically sorted
                check non-empty name for tensor initializers
                check duplicates in graph.inputs, graph.initializer, graph.sparse_initializer, (potentially, if all node-checks pass) graph.outputs
            """
            # NOTE: `onnx.checker.check_model` checks Field `shape` of `type` in `ValueInfoProto` of graph.
            # But graphsurgeon doesn't add this field when exporting from ONNX to GS.
            # So we need to manually check and add that field so `onnx.checker` is happy.
            for output in onnx_model.graph.output:
                if not output.type.tensor_type.HasField("shape"):
                    output.type.tensor_type.shape.dim.add()

            # handle large models
            if onnx_model.ByteSize() > Lint.MAXIMUM_PROTOBUF:
                checker_input = self.arg_groups[ModelArgs].path
                G_LOGGER.warning(
                    "Given ONNX Model >2GB. ONNX-Checker will run with model path as input instead.\n"
                    "NOTE: The external data needs to be under the same directory as model path."
                )
            else:
                checker_input = onnx_model

            try:
                onnx.checker.check_model(checker_input)
            except onnx.checker.ValidationError as err:
                if Lint.ONNX_CHECKER_IGNORE_SUBSTR not in str(err):
                    self.report.add(
                        level=Lint.Level.EXCEPTION,
                        message=str(err),
                        source=Lint.Source.ONNX_CHECKER,
                    )
                return False

            return True

        def capture(
            func,
        ):
            """
            Decorator to capture stdout, exceptions, and warnings from the `ort_inference_check` function.
            Uses C-level stdout and stderr redirection to capture any warnings printed by the ONNX Runtime.
            Note: This is not thread-safe!
            """
            # The stdout redirector code was generalized from a post on Eli Bendersky's website.
            # The original code for POSIX-specific systems can be found at https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/.

            @contextlib.contextmanager
            def stderr_redirector(stream):
                # The original fd stderr points to. Usually 2 on POSIX systems.
                original_stderr_fd = sys.stderr.fileno()

                def _redirect_stderr(to_fd):
                    """Redirect stderr to the given file descriptor."""
                    # Flush and close sys.stderr - also closes the file descriptor
                    sys.stderr.close()
                    # Make original_stderr_fd point to the same file as to_fd
                    os.dup2(to_fd, original_stderr_fd)
                    # Create a new sys.stderr that points to the redirected fd
                    sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, "wb"))

                # Save a copy of the original stderr fd in saved_stderr_fd
                saved_stderr_fd = os.dup(original_stderr_fd)
                try:
                    # Create a temporary file and redirect stderr to it
                    tfile = tempfile.TemporaryFile(mode="w+b")
                    _redirect_stderr(tfile.fileno())
                    # Yield to caller, then redirect stderr back to the saved fd
                    yield
                    _redirect_stderr(saved_stderr_fd)
                    # Copy contents of temporary file to the given stream
                    tfile.flush()
                    tfile.seek(0, io.SEEK_SET)
                    stream.write(tfile.read())
                finally:
                    tfile.close()
                    os.close(saved_stderr_fd)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                captured_stdout = io.StringIO()
                captured_stderr = io.BytesIO()
                captured_exception = None
                result = None
                with contextlib.redirect_stdout(captured_stdout), stderr_redirector(
                    captured_stderr
                ):
                    try:
                        # Execute the function
                        result = func(*args, **kwargs)
                    except Exception as err:  # pylint: disable=broad-except
                        captured_exception = err
                UTF_TYPE = "utf-16-le" if os.name == "nt" else "utf-8"
                stderr_msg = captured_stderr.getvalue().decode(
                    UTF_TYPE
                )  # platform-dependent
                stdout_msg = captured_stdout.getvalue()
                return (result, captured_exception, stderr_msg, stdout_msg)

            return wrapper

        @capture
        def _ort_inference_check(
            model_bytes: Union[bytes, str], feed_dict: OrderedDict
        ) -> Optional[OrderedDict]:
            """
            Runs inference using ONNX-Runtime.

            Args:
                model_bytes (Union[bytes, str]): The model bytes or path for the model.
                feed_dict (dict): The feed dictionary to use for inference.

            Returns:
                dict: The output dictionary from the inference run.
                      None if the inference run fails.

            NOTE: This function is decorated with `capture` to capture all stdout, stderr, and exceptions.
            """
            with onnxrt_backend.OnnxrtRunner(
                self.arg_groups[OnnxrtSessionArgs].load_onnxrt_session(model_bytes)
            ) as runner:
                output_dict = runner.infer(feed_dict)
            return output_dict

        @capture
        def _unused_info_helper(graph: "gs.Graph"):
            """
            Helper function to report unused nodes and input tensors in the graph.
            Calls `graph.cleanup()` in-place to remove unused nodes and input tensors.

            Returns the difference between the original and cleaned graph as a tuple of sets

            NOTE: This function is decorated with `capture` to capture all stdout, stderr, and exceptions.
            """

            orig_input_tensor_names = {inp.name for inp in graph.inputs}
            orig_node_info = {(node.name, node.op) for node in graph.nodes}

            ### in-place clean ###
            # remove any tensor or node that doesn't contribute to the output
            # NOTE: only the top-level nodes are removed, and not the subgraphs.
            graph.cleanup(recurse_subgraphs=False, remove_unused_graph_inputs=True)

            cleaned_input_tensor_names = {inp.name for inp in graph.inputs}
            cleaned_node_info = {(node.name, node.op) for node in graph.nodes}

            return (
                orig_node_info - cleaned_node_info,
                orig_input_tensor_names - cleaned_input_tensor_names,
            )

        def _report_unused_info(graph: "gs.Graph"):
            """
            Checks for unused nodes and inputs in the graph.
            Appends to the report as warnings.

            Args:
                graph (gs.Graph): The graph to check for unused nodes and tensors.

            Note:
                - This function avoids copying the graph to avoid memory overhead,
                    and instead modifies the graph in-place by calling `graph.cleanup()`.
                    Therefore, this function is intentionally called at the end of the linting process.
                - All nodes in the graph are expected to have non-empty names.
            """

            (unused_node_info, unused_input_tensor_names), exception, _, _ = (
                _unused_info_helper(graph)
            )

            if exception:
                # something went wrong here.
                G_LOGGER.internal_error(
                    f"Failed to report unused nodes. Error: {exception}"
                )
                G_LOGGER.warning(
                    f"Failed to report unused nodes. Error: {exception}. Continuing..."
                )

            # report unused tensors that are also inputs (intermediate tensors are not reported)
            for inp_name in sorted(list(unused_input_tensor_names)):
                self.report.add(
                    Lint.Level.WARNING,
                    Lint.Source.ONNX_GS,
                    f"Input: '{inp_name}' does not affect outputs, can be removed.",
                )

            # report unused nodes of the outermost graph
            for node_name, op in sorted(list(unused_node_info), key=lambda x: x[0]):
                self.report.add(
                    Lint.Level.WARNING,
                    Lint.Source.ONNX_GS,
                    "Does not affect outputs, can be removed.",
                    node_name=node_name,
                    op=op,
                )

        # instantiate the report
        self.report = Lint.Report()

        # tries to load the model in-memory using OnnxLoadArgs.load_onnx()
        # TODO: find a way to avoid loading the whole model in memory if not-required
        # Currently we'll need the model for the following reasons:
        # 1. to calculate the model size
        # 2. to obtain input shape metadata for generating feed dict
        # 3. if OnnxLoadArgs.must_use_onnx_loader() is True.
        # 4. overriding input shapes if provided by the user
        onnx_model = self.load_helper(args)
        # handle invalid or empty onnx model
        if not onnx_model:
            return 1  # invalid
        if len(onnx_model.graph.node) == 0:
            self.report.add(
                Lint.Level.WARNING,
                Lint.Source.ONNX_LOADER,
                "ONNX model has no nodes",
            )
            self.report.export(args.output)
            return 0  # empty

        graph = gs.import_onnx(onnx_model)

        ### Preprocess graph ###
        # override input shapes if provided by the user
        user_input_metadata = self.arg_groups[ModelArgs].input_shapes
        if user_input_metadata:
            graph = tools_util.override_input_shapes(graph, user_input_metadata)
        G_LOGGER.verbose("ONNX Model loaded into linter succesfully.")
        # rename any nodes with empty names
        _handle_empty_names(graph)

        onnx_model = gs.export_onnx(graph, do_type_check=False)  # update the ONNX model

        ### report.add(Duplicate node names violation) ###
        _duplicate_node_name_check(graph)

        ### report.add(ONNX spec violations) ###
        is_onnx_check_passing = _onnx_spec_check(onnx_model)
        G_LOGGER.verbose(f"ONNX Checker passed: {is_onnx_check_passing}")

        # prepare feed_dict for `ort_inference_check`, that will later be re-used.
        feed_dict = self.prepare_feed_dict(onnx_model)

        if is_onnx_check_passing:
            ### full ORT inference as preliminary check ###
            model_bytes = None  # so that model is picked from `self.arg_groups`
            _, exception, warn_str, _ = _ort_inference_check(model_bytes, feed_dict)
            # NOTE: we ignore stdout as it contains info from polygraphy not relevant to linting.

            if not exception:
                # ORT inference check passes, early exit
                # any recorded warnings from stderr are added to the report.
                # NOTE: This is only done if early-exiting, as otherwise these warnings tend to be repeats
                # of node level warnings/exceptions.
                if warn_str:
                    warnings = warn_str.split("\n")
                    for warning in warnings:
                        if len(warning) > 0:
                            self.report.add(
                                Lint.Level.WARNING,
                                Lint.Source.ONNXRUNTIME,
                                warning,
                            )

                ### report.add(unused nodes and tensors) ###
                _report_unused_info(graph)

                self.report.summary["passing"] = {node.name for node in graph.nodes}
                self.report.export(args.output)
                G_LOGGER.verbose(
                    "ORT inference check passed. Model is valid. Early exiting."
                )
                return 0
            if isinstance(exception, PolygraphyException):
                # PolygraphyException is raised when the provided input is not compatible with polygraphy
                # This could be due to improper input, unsupported provider etc. that user needs to fix.
                # This is not raised due to errors in ONNX Model, so we shouldn't handle it.
                G_LOGGER.critical(f"PolygraphyException: {exception}")
                raise exception
            G_LOGGER.verbose(f"ORT inference check failed with error: '{exception}'")

        # start Node-level linting
        with Lint.ContextManager(graph) as lcm:
            lcm.set_graph_inputs(
                feed_dict
            )  # load the cache with initial feed_dict values for iterative inference.

            for _ in lcm.nodes():
                g = lcm.make_singleton_graph()
                inference_output = None

                if g:  # has valid ancestors. Can perform inference.
                    model_bytes = onnx_backend.BytesFromOnnx(
                        gs.export_onnx(g, do_type_check=False)
                    )
                    inference_output, exception, _, _ = _ort_inference_check(
                        model_bytes, lcm.feed_dict()
                    )
                    # NOTE: we ignore stdout and stderr as it contains info from polygraphy not relevant to linting.
                    err_str = str(exception) if exception else ""
                    if any(
                        [
                            substr in err_str
                            for substr in Lint.CUSTOM_OP_EXCEPTION_SUBSTRS
                        ]
                    ):
                        self.report.add(
                            level=Lint.Level.WARNING,
                            source=Lint.Source.ONNXRUNTIME,
                            message=err_str,
                            node_name=g.name,
                            op=g.nodes[0].op,
                        )
                    else:
                        self.report.add(
                            level=Lint.Level.EXCEPTION,
                            source=Lint.Source.ONNXRUNTIME,
                            message=err_str,
                            node_name=g.name,
                            op=g.nodes[0].op,
                        )

                # update : cache new outputs if any, and remove stale tensors from cache.
                lcm.update(inference_output)

            ### report.add(unused nodes and tensors) ###
            _report_unused_info(graph)
            self.report.export(args.output)

            return int(not self.report.is_model_valid)
