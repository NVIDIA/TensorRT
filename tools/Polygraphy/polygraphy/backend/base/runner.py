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
import copy
import time
from collections import defaultdict

from polygraphy import config, func, mod, util
from polygraphy.logger import G_LOGGER, LogMode

np = mod.lazy_import("numpy")


@mod.export()
class BaseRunner:
    """
    Base class for Polygraphy runners. All runners should override the functions and attributes specified here.
    """

    RUNNER_COUNTS = defaultdict(int)

    def __init__(self, name=None, prefix=None):
        """
        Args:
            name (str):
                    The name to use for this runner.
            prefix (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
                    Only used if name is not provided.
        """
        prefix = util.default(prefix, "Runner")
        if name is None:
            count = BaseRunner.RUNNER_COUNTS[prefix]
            BaseRunner.RUNNER_COUNTS[prefix] += 1
            name = f"{prefix}-N{count}-{time.strftime('%x')}-{time.strftime('%X')}"
        self.name = name
        self.inference_time = None

        self.is_active = False
        """bool: Whether this runner has been activated, either via context manager, or by calling ``activate()``."""

    def __enter__(self):
        """
        Activate the runner for inference. For example, this may involve allocating CPU or GPU memory.
        """
        self.activate()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Deactivate the runner. For example, this may involve freeing CPU or GPU memory.
        """
        self.deactivate()

    # Implementation for runner activation. Derived classes should override this function
    # rather than ``activate()``.
    def activate_impl(self):
        pass

    def activate(self):
        """
        Activate the runner for inference. For example, this may involve allocating CPU or GPU memory.

        Generally, you should use a context manager instead of manually activating and deactivating.
        For example:
        ::

            with RunnerType(...) as runner:
                runner.infer(...)
        """
        if self.is_active:
            G_LOGGER.warning(
                f"{self.name:35} | Already active; will not activate again. "
                "If you really want to activate this runner again, call activate_impl() directly"
            )
            return

        if config.INTERNAL_CORRECTNESS_CHECKS:
            self._pre_activate_runner_state = copy.copy(vars(self))

        self.activate_impl()
        self.is_active = True

    def get_input_metadata_impl(self):
        """
        Implemenation for `get_input_metadata`. Derived classes should override this function
        rather than `get_input_metadata`.
        """
        raise NotImplementedError("BaseRunner is an abstract class")

    @func.constantmethod
    def get_input_metadata(self):
        """
        Returns information about the inputs of the model.
        Shapes here may include dynamic dimensions, represented by ``None``.
        Must be called only after ``activate()`` and before ``deactivate()``.

        Returns:
            TensorMetadata: Input names, shapes, and data types.
        """
        return self.get_input_metadata_impl()

    # Implementation for runner inference. Derived classes should override this function
    # rather than ``infer()``
    def infer_impl(self, feed_dict):
        raise NotImplementedError("BaseRunner is an abstract class")

    def infer(self, feed_dict, check_inputs=True, *args, **kwargs):
        """
        Runs inference using the provided feed_dict.

        Must be called only after ``activate()`` and before ``deactivate()``.

        NOTE: Some runners may accept additional parameters in infer().
        For details on these, see the documentation for their `infer_impl()` methods.

        Args:
            feed_dict (OrderedDict[str, numpy.ndarray]):
                    A mapping of input tensor names to corresponding input NumPy arrays.

            check_inputs (bool):
                    Whether to check that the provided ``feed_dict`` includes the expected inputs
                    with the expected data types and shapes.
                    Disabling this may improve performance.
                    Defaults to True.

        Attributes:
            inference_time (float):
                    The time required to run inference in seconds.
                    Derived classes should set this so that performance metrics are accurate.

        Returns:
            OrderedDict[str, numpy.ndarray]:
                    A mapping of output tensor names to their corresponding NumPy arrays.

                    IMPORTANT: Runners may reuse these output buffers. Thus, if you need to save
                    outputs from multiple inferences, you should make a copy with ``copy.deepcopy(outputs)``.
        """
        if not self.is_active:
            G_LOGGER.critical(f"{self.name:35} | Must be activated prior to calling infer()")

        if check_inputs:
            input_metadata = self.get_input_metadata()
            G_LOGGER.verbose(f"Runner input metadata is: {input_metadata}")

            util.check_sequence_contains(feed_dict.keys(), input_metadata.keys(), name="feed_dict", items_name="inputs")

            for name, inp in feed_dict.items():
                meta = input_metadata[name]
                if not np.issubdtype(inp.dtype, meta.dtype):
                    G_LOGGER.critical(
                        f"Input tensor: {name} | Received unexpected dtype: {inp.dtype}.\nNote: Expected type: {meta.dtype}"
                    )

                if not util.is_valid_shape_override(inp.shape, meta.shape):
                    G_LOGGER.critical(
                        f"Input tensor: {name} | Received incompatible shape: {inp.shape}.\nNote: Expected a shape compatible with: {meta.shape}"
                    )

        return self.infer_impl(feed_dict, *args, **kwargs)

    @func.constantmethod
    def last_inference_time(self):
        """
        Returns the total inference time in seconds required during the last call to ``infer()``.

        Must be called only after ``activate()`` and before ``deactivate()``.

        Returns:
            float: The time in seconds, or None if runtime was not measured by the runner.
        """
        if self.inference_time is None:
            msg = f"{self.name:35} | `inference_time` was not set. Inference time will be incorrect! "
            msg += "To correctly compare runtimes, please set the `inference_time` attribute in `infer_impl()`"

            G_LOGGER.internal_error(msg)
            G_LOGGER.warning(msg, mode=LogMode.ONCE)
            return None
        return self.inference_time

    # Implementation for runner deactivation. Derived classes should override this function
    # rather than ``deactivate()``.
    def deactivate_impl(self):
        pass

    def deactivate(self):
        """
        Deactivate the runner. For example, this may involve freeing CPU or GPU memory.

        Generally, you should use a context manager instead of manually activating and deactivating.
        For example:
        ::

            with RunnerType(...) as runner:
                runner.infer(...)
        """
        if not self.is_active:
            G_LOGGER.warning(
                f"{self.name:35} | Not active; will not deactivate. If you really want to deactivate this runner, call deactivate_impl() directly"
            )
            return

        self.inference_time = None
        self.is_active = None

        self.deactivate_impl()
        self.is_active = False
        if config.INTERNAL_CORRECTNESS_CHECKS:
            old_state = self._pre_activate_runner_state
            del self._pre_activate_runner_state
            if old_state != vars(self):
                G_LOGGER.internal_error(
                    f"Runner state was not reset after deactivation. Note:\nOld state: {old_state}\nNew state: {vars(self)}"
                )

    def __del__(self):
        if self.is_active:
            # __del__ is not guaranteed to be called, but when it is, this could be a useful warning.
            print(f"[W] {self.name:35} | Was activated but never deactivated. This could cause a memory leak!")
