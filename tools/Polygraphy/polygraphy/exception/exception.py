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

from polygraphy import mod


# Do not raise this exception manually. Instead, use G_LOGGER.critical().
@mod.export()
class PolygraphyException(Exception):
    """
    An exception raised by Polygraphy.
    """

    pass


# Do not raise this exception manually. Instead, use G_LOGGER.internal_error().
@mod.export()
class PolygraphyInternalException(Exception):
    """
    An exception raised when a Polygraphy internal check is violated.
    Polygraphy internal checks can be enabled by setting the ``POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS``
    environment variable to ``1``.
    This is *not* a child class of PolygraphyException because it
    indicates a bug in Polygraphy itself.
    """

    pass
