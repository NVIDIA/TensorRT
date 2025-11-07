#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from polygraphy import config as polygraphy_config, mod, util
from polygraphy.backend.trt.config import _CreateConfigCommon
from polygraphy.backend.trt.util import inherit_and_extend_docstring
from polygraphy.logger import G_LOGGER
from polygraphy.mod.trt_importer import lazy_import_trt

trt = lazy_import_trt()


@mod.export(funcify=True, func_name="create_config_rtx")
class CreateConfigRTX(_CreateConfigCommon):
    """
    Functor that creates an IBuilderConfig with TensorRT-RTX specific features.
    """

    @inherit_and_extend_docstring(_CreateConfigCommon.__init__)
    def __init__(
        self,
        use_gpu=None,
        compute_capabilities=None,
        **kwargs
    ):
        """
        Creates an IBuilderConfig with TensorRT-RTX specific features.

        Args:
            use_gpu (bool):
                    Whether to use the current GPU device as target for engine compilation.
                    Equivalent to setting ComputeCapability.CURRENT. This is mutually exclusive with compute_capabilities.
                    Defaults to False.
            compute_capabilities (List[Tuple[int, int]]):
                    List of (major, minor) compute capability tuples to target for engine compilation.
                    This is mutually exclusive with use_gpu. When specified, the engine can only run on devices
                    with the specified compute capabilities.
                    Defaults to None.
        """
        super().__init__(**kwargs)
        self.use_gpu = util.default(use_gpu, False)
        self.compute_capabilities = compute_capabilities

        if self.use_gpu and self.compute_capabilities:
            G_LOGGER.critical("use_gpu and compute_capabilities are mutually exclusive.")

        self._validator()

    def _validator(self):
        """
        Validates initialization parameters for TensorRT-RTX specific features.
        """
        if self.use_gpu or self.compute_capabilities is not None:
            if not polygraphy_config.USE_TENSORRT_RTX:
                G_LOGGER.critical("--compute-capabilities and --use-gpu settings are only supported with USE_TENSORRT_RTX=1.")
            
            # Validate compute capabilities format and availability
            if self.compute_capabilities:
                for major, minor in self.compute_capabilities:
                    cap_name = f"SM{major}{minor}"
                    if not hasattr(trt.ComputeCapability, cap_name):
                        G_LOGGER.critical(f"Compute capability {major}.{minor} ({cap_name})"
                                           " not supported by this TensorRT-RTX version.")

    def _configure_flags(self, builder, network, config):
        """
        Validates and configures TensorRT-RTX-specific features.

        Args:
            builder (trt.Builder): The TensorRT builder
            network (trt.INetworkDefinition): The TensorRT network
            config (trt.IBuilderConfig): The TensorRT builder config to modify
        """
        # Set compute capabilities if specified
        if self.use_gpu or self.compute_capabilities is not None:
            try:
                if self.use_gpu:
                    # Use current GPU device
                    config.num_compute_capabilities = 1
                    config.set_compute_capability(trt.ComputeCapability.CURRENT, 0)
                    G_LOGGER.info("Using current GPU device for engine compilation (ComputeCapability.CURRENT)")
                elif self.compute_capabilities:
                    # Set specific compute capabilities
                    config.num_compute_capabilities = len(self.compute_capabilities)
                    G_LOGGER.info(f"Setting {len(self.compute_capabilities)} target compute capabilities: {self.compute_capabilities}")
                    for i, (major, minor) in enumerate(self.compute_capabilities):
                        cap_name = f"SM{major}{minor}"
                        compute_cap = getattr(trt.ComputeCapability, cap_name)
                        config.set_compute_capability(compute_cap, i)
            except Exception as e:
                G_LOGGER.critical(f"Failed to set compute capabilities: {e}. You are likely not using a TensorRT-RTX build.")

    @util.check_called_by("__call__")
    def call_impl(self, builder, network):
        """
        Callable implementation that creates and configures the IBuilderConfig with TensorRT-RTX features.
        """
        # Enable all common config options
        config = super().call_impl(builder, network)

        self._configure_flags(builder, network, config)

        return config
