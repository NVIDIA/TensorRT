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

from polygraphy import func, mod


@mod.export()
class BaseLoader:
    """
    Base class for Polygraphy Loaders.
    """

    def call_impl(self, *args, **kwargs):
        """
        Implementation for ``__call__``. Derived classes should implement this
        method rather than ``__call__``.
        """
        raise NotImplementedError("BaseLoader is an abstract class")

    @func.constantmethod
    def __call__(self, *args, **kwargs):
        """
        Invokes the loader by forwarding arguments to ``call_impl``.

        Note: ``call_impl`` should *not* be called directly - use this function instead.
        """
        __doc__ = self.call_impl.__doc__
        return self.call_impl(*args, **kwargs)
