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

from polygraphy import mod, util
from polygraphy.logger import G_LOGGER
from polygraphy.mod.trt_importer import lazy_import_trt
from polygraphy.tools.args import (
    DataLoaderArgs,
    ModelArgs,
    OnnxInferShapesArgs,
    OnnxLoadArgs,
    TrtConfigArgs,
    TrtLoadEngineBytesArgs,
    TrtLoadNetworkArgs,
    TrtLoadPluginsArgs,
    TrtOnnxFlagArgs,
    TrtSaveEngineBytesArgs,
)
from polygraphy.tools.base import Tool
from polygraphy.tools.debug.subtool.iterative_debug_args import (
    ArtifactSortArgs,
    CheckCmdArgs,
    IterativeDebugArgs,
)

trt_backend = mod.lazy_import("polygraphy.backend.trt")
trt = lazy_import_trt()


class BaseCheckerSubtool(Tool):
    def __init__(
        self,
        name,
        precision_constraints_default=None,
        allow_no_artifacts_warning=True,
        allow_until_opt=None,
        allow_debug_replay=None,
    ):
        super().__init__(name)
        self._precision_constraints_default = precision_constraints_default
        self._allow_no_artifacts_warning = allow_no_artifacts_warning
        self._allow_until_opt = allow_until_opt
        self._allow_debug_replay = allow_debug_replay

    def get_subscriptions_impl(self):
        return [
            CheckCmdArgs(),
            ArtifactSortArgs(
                allow_no_artifacts_warning=self._allow_no_artifacts_warning
            ),
            IterativeDebugArgs(
                iter_art_opt_default="polygraphy_debug.engine",
                allow_until_opt=self._allow_until_opt,
                allow_debug_replay=self._allow_debug_replay,
            ),
            ModelArgs(model_opt_required=True, input_shapes_opt_name=False),
            OnnxInferShapesArgs(),
            OnnxLoadArgs(outputs_opt_prefix=False),
            DataLoaderArgs(),  # For int8 calibration
            TrtConfigArgs(
                precision_constraints_default=self._precision_constraints_default
            ),
            TrtLoadPluginsArgs(),
            TrtOnnxFlagArgs(),
            TrtLoadNetworkArgs(),
            TrtLoadEngineBytesArgs(),
            TrtSaveEngineBytesArgs(output_opt=False),
        ]

    def show_start_end_logging_impl(self, args):
        return True

    def setup(self, args, network):
        """
        Initialize a subtool.
        """
        pass

    def step(self, success):
        """
        Advances the iterator and returns whether to stop iteration.

        Args:
            success (bool): Whether the check command succeeded (True) or failed (False).

        Returns:
            bool: Whether to stop iteration.
        """
        raise NotImplementedError("Must be implemented by child classes!")

    def process_network(self, network):
        """
        Process the TensorRT network prior to engine building.

        Args:
            network (trt.INetworkDefinition): The network to process.
        """
        pass

    def remaining(self):
        """
        Returns the estimated number of iterations remaining.
        """
        pass

    def run_impl(self, args):
        builder, network, _ = util.unpack_args(
            self.arg_groups[TrtLoadNetworkArgs].load_network(), 3
        )

        self.setup(args, network)

        def make_iter_art(_):
            self.process_network(network)

            try:
                serialized_engine = self.arg_groups[
                    TrtLoadEngineBytesArgs
                ].load_engine_bytes((builder, network))
            except Exception as err:
                G_LOGGER.warning(
                    f"Failed to create network or engine, continuing to the next iteration.\nNote: Error was: {err}"
                )
                G_LOGGER.internal_error(
                    "Failed to create network or engine. See warning above for details."
                )
                self.arg_groups[IterativeDebugArgs].skip_iteration(success=False)
            else:
                # Don't need to keep the engine around in memory - just serialize to disk and free it.
                with serialized_engine:
                    self.arg_groups[TrtSaveEngineBytesArgs].save_engine_bytes(
                        serialized_engine,
                        self.arg_groups[IterativeDebugArgs].iter_artifact_path,
                    )

        def advance(context):
            if self.step(context.success):
                self.arg_groups[IterativeDebugArgs].stop_iteration()

        self.arg_groups[IterativeDebugArgs].iterate(
            make_iter_art_func=make_iter_art,
            advance_func=advance if not self._allow_until_opt else None,
            get_remaining_func=lambda: self.remaining(),
        )
