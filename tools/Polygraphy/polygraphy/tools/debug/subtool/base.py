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
import contextlib
import os

from polygraphy import mod, util
from polygraphy.logger import G_LOGGER
from polygraphy.tools.args import (
    DataLoaderArgs,
    ModelArgs,
    OnnxLoaderArgs,
    OnnxShapeInferenceArgs,
    TrtConfigArgs,
    TrtEngineLoaderArgs,
    TrtEngineSaveArgs,
    TrtNetworkLoaderArgs,
    TrtPluginLoaderArgs,
)
from polygraphy.tools.base import Tool
from polygraphy.tools.debug.subtool.artifact_sorter import ArtifactSorterArgs

trt_backend = mod.lazy_import("polygraphy.backend.trt")


class BaseCheckerSubtool(Tool):
    def __init__(self, name, strict_types_default=None, prefer_artifacts=True):
        super().__init__(name)
        self.subscribe_args(ArtifactSorterArgs("polygraphy_debug.engine", prefer_artifacts=prefer_artifacts))
        self.subscribe_args(ModelArgs(model_required=True, inputs=None))
        self.subscribe_args(OnnxShapeInferenceArgs())
        self.subscribe_args(OnnxLoaderArgs(output_prefix=None))
        self.subscribe_args(DataLoaderArgs())  # For int8 calibration
        self.subscribe_args(TrtConfigArgs(strict_types_default=strict_types_default))
        self.subscribe_args(TrtPluginLoaderArgs())
        self.subscribe_args(TrtNetworkLoaderArgs())
        self.subscribe_args(TrtEngineLoaderArgs())
        self.subscribe_args(TrtEngineSaveArgs(output=False))

    def setup(self, args, network):
        """
        Initialize a subtool.
        """
        pass

    def stop(self, iteration, success):
        """
        Controls when to stop iteration.

        Args:
            iteration (int): The current iteration, starting at 0.
            success (bool): Whether the check command succeeded (True) or failed (False).

        Returns:
            bool: Whether to stop iteration.
        """
        raise NotImplementedError("Must be implemented by child classes!")

    def process_network(self, network, prev_success):
        """
        Process the TensorRT network prior to engine building.

        Args:
            network (trt.INetworkDefinition): The network to process.
            prev_success (bool):
                Whether the previous iteration succeeded.
                This value is always True for the 0th iteration.
        """
        pass

    def remaining(self):
        """
        Returns the estimated number of iterations remaining.
        """
        pass

    def run(self, args):
        G_LOGGER.start("Starting iterations")

        builder, network, parser = util.unpack_args(self.arg_groups[TrtNetworkLoaderArgs].load_network(), 3)

        with contextlib.ExitStack() as stack:
            stack.enter_context(builder)
            stack.enter_context(network)
            if parser:
                stack.enter_context(parser)

            self.setup(args, network)

            num_passed = 0
            num_total = 0

            success = True
            MAX_COUNT = 100000  # We don't want to loop forever. This many iterations ought to be enough for anybody.
            for iteration in range(MAX_COUNT):
                remaining = self.remaining()
                G_LOGGER.start(
                    "RUNNING | Iteration {:}{:}".format(
                        iteration + 1,
                        " | Approximately {:} iteration(s) remaining".format(remaining)
                        if remaining is not None
                        else "",
                    )
                )

                self.process_network(network, success)

                try:
                    engine = self.arg_groups[TrtEngineLoaderArgs].build_engine((builder, network))
                except Exception as err:
                    G_LOGGER.warning(
                        "Failed to create network or engine, continuing to the next iteration.\n"
                        "Note: Error was: {:}".format(err)
                    )
                    G_LOGGER.internal_error("Failed to create network or engine. See warning above for details.")
                    success = False
                else:
                    # Don't need to keep the engine around in memory - just serialize to disk and free it.
                    with engine:
                        self.arg_groups[TrtEngineSaveArgs].save_engine(
                            engine, self.arg_groups[ArtifactSorterArgs].iter_artifact
                        )
                    success = self.arg_groups[ArtifactSorterArgs].sort_artifacts(iteration + 1)

                num_total += 1
                if success:
                    num_passed += 1

                if self.stop(iteration, success):
                    break
            else:
                G_LOGGER.warning(
                    "Maximum number of iterations reached: {:}.\n"
                    "Iteration has been halted to prevent an infinite loop!".format(MAX_COUNT)
                )

        G_LOGGER.finish(
            "Finished {:} iteration(s) | Passed: {:}/{:} | Pass Rate: {:}%".format(
                iteration + 1, num_passed, num_total, float(num_passed) * 100 / float(num_total)
            )
        )
