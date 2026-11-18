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


class OnnxGraphSurgeonException(Exception):
    """
    An exception raised by ONNX-GraphSurgeon.
    """

    pass


class OnnxGraphSurgeonResourceError(OnnxGraphSurgeonException, RuntimeError):
    """
    Raised when an operation would consume an unreasonable amount of resources,
    such as densifying a sparse tensor whose declared dense shape is far larger
    than the data it actually stores.

    It also derives from the built-in ``RuntimeError`` so that callers can guard
    against resource-exhaustion attempts without importing ONNX-GraphSurgeon.
    """

    pass
