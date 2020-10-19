#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
from polygraphy.util import misc
from polygraphy.logger.logger import G_LOGGER, LogMode

from collections import defaultdict
import time


class BaseRunner(object):
    """
    The base class for runner objects. All runners should override the functions and attributes specified here.
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
        prefix = misc.default_value(prefix, "Runner")
        if name is None:
            count = BaseRunner.RUNNER_COUNTS[prefix]
            BaseRunner.RUNNER_COUNTS[prefix] += 1
            name = "{:}-N{:}-{:}-{:}".format(prefix, count, time.strftime("%x"), time.strftime("%X"))
        self.name = name
        self.inference_time = None

        self.is_active = False
        """bool: Whether this runner has been activated, either via context manager, or by calling ``activate()``."""


    def last_inference_time(self):
        """
        Returns the total time required for the last call to ``infer()``.

        Returns:
            float: The time in seconds, or None if runtime was not measured by the runner.
        """
        if self.inference_time is None:
            G_LOGGER.warning("inference_time was not set by this runner. Inference time will be incorrect!"
                             "To correctly compare runtimes, please set the inference_time property in the"
                             "infer() function", mode=LogMode.ONCE)
            return None
        return self.inference_time


    def __enter__(self):
        self.activate()
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.deactivate()


    def activate_impl(self):
        """
        Implementation for runner activation. Derived classes should override this function
        rather than ``activate()``.
        """
        pass


    def activate(self):
        """
        Activate the runner for inference. This may involve allocating GPU buffers, for example.
        """
        if self.is_active:
            G_LOGGER.warning("Runner is already active; Will not activate again. If you really want to "
                              "activate this runner twice, call activate_impl() directly")
            return

        try:
            self.activate_impl()
        finally:
            self.is_active = True


    def infer(self, feed_dict):
        """
        Runs inference using the provided feed_dict.

        Args:
            feed_dict (OrderedDict[str, np.ndarray]): A mapping of input tensor names to corresponding input NumPy arrays.

        Returns:
            OrderedDict[str, np.ndarray]:
                    A mapping of output tensor names to their corresponding NumPy arrays.
                    IMPORTANT: Runners may reuse these output buffers. Thus, if you need to save
                    outputs from multiple inferences, you should make a copy with ``copy.copy(outputs)``.
        """
        raise NotImplementedError("BaseRunner is an abstract class")


    def get_input_metadata(self):
        """
        Returns information about the inputs of the model.
        Shapes here may include dynamic dimensions, represented by ``None``.
        Must be called only after activate() and before deactivate().

        Returns:
            TensorMetadata: Input names, shapes, and data types.
        """
        raise NotImplementedError("BaseRunner is an abstract class")


    def deactivate_impl(self):
        """
        Implementation for runner deactivation. Derived classes should override this function
        rather than ``deactivate()``.
        """
        pass


    def deactivate(self):
        """
        Deactivate the runner.
        """
        if not self.is_active:
            G_LOGGER.warning("Runner is not active; Will not deactivate. If you really want to "
                                "deactivate this runner, call deactivate_impl() directly")
            return

        try:
            self.deactivate_impl()
        finally:
            self.is_active = False
