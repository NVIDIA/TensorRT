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
import copy
import time
from collections import defaultdict

from polygraphy import config, func, mod, util
from polygraphy.datatype import DataType
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy.backend.base import util as base_util

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

        Derived classes may return any kind of data type supported by Polygraphy's DataType
        class (e.g. np.dtype, torch.dtype, etc.)
        """
        raise NotImplementedError("BaseRunner is an abstract class")

    @func.constantmethod
    def get_input_metadata(self, use_numpy_dtypes=None):
        """
        Returns information about the inputs of the model.
        Shapes here may include dynamic dimensions, represented by ``None``.
        Must be called only after ``activate()`` and before ``deactivate()``.

        Args:
            use_numpy_dtypes (bool):
                    [DEPRECATED] Whether to return NumPy data types instead of Polygraphy ``DataType`` s.
                    This is provided to retain backwards compatibility. In the future,
                    this parameter will be removed and Polygraphy ``DataType`` s will
                    always be returned. These can be converted to NumPy data types by calling the `numpy()` method.
                    Defaults to True.

        Returns:
            TensorMetadata: Input names, shapes, and data types.
        """
        if not self.is_active:
            G_LOGGER.critical(
                f"{self.name:35} | Must be activated prior to calling get_input_metadata()"
            )

        use_numpy_dtypes = util.default(use_numpy_dtypes, True)

        meta = self.get_input_metadata_impl()

        for name, (dtype, _) in meta.items():
            dtype = DataType.from_dtype(dtype)
            if use_numpy_dtypes:
                mod.warn_deprecated(
                    "Returning NumPy data types instead of Polygraphy `DataType`s from `get_input_metadata()`",
                    use_instead=None,
                    remove_in="0.60.0",
                )
                meta[name]._dtype = DataType.to_dtype(dtype, "numpy")
        return meta

    # Implementation for runner inference. Derived classes should override this function
    # rather than ``infer()``
    # Derived classes should also set the `inference_time` property so that performance metrics are accurate.
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

        Returns:
            OrderedDict[str, numpy.ndarray]:
                    A mapping of output tensor names to their corresponding NumPy arrays.

                    IMPORTANT: Runners may reuse these output buffers. Thus, if you need to save
                    outputs from multiple inferences, you should make a copy with ``copy.deepcopy(outputs)``.
        """
        if not self.is_active:
            G_LOGGER.critical(
                f"{self.name:35} | Must be activated prior to calling infer()"
            )

        if check_inputs:
            input_metadata = self.get_input_metadata(use_numpy_dtypes=False)
            G_LOGGER.verbose(
                f"{self.name:35} | Input metadata is: {input_metadata}",
                mode=LogMode.ONCE,
            )
            base_util.check_inputs(feed_dict, input_metadata)

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
            print(
                f"[W] {self.name:35} | Was activated but never deactivated. This could cause a memory leak!"
            )
