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

from polygraphy import util
from polygraphy.logger import G_LOGGER


def check_inputs(feed_dict, input_metadata):
    """
    Checks the provided `feed_dict` against expected input metadata.

    Args:
        feed_dict (Dict[str, Union[DeviceView, numpy.ndarray, torch.Tensor]]):
                A mapping of input names to arrays.
        input_metadata (TensorMetadata):
                The expected input metadata.
    """
    util.check_sequence_contains(
        feed_dict.keys(), input_metadata.keys(), name="input data", items_name="inputs"
    )

    for name, inp in feed_dict.items():
        meta = input_metadata[name]

        # The "buffer" might just be a pointer, in which case we can't do any further checks with it, so we skip it.
        if isinstance(inp, int):
            continue

        dtype = util.array.dtype(inp)
        if dtype != meta.dtype:
            G_LOGGER.critical(
                f"Input tensor: {name} | Received unexpected dtype: {dtype}.\nNote: Expected type: {meta.dtype}"
            )

        shape = util.array.shape(inp)
        if not util.is_valid_shape_override(shape, meta.shape):
            G_LOGGER.critical(
                f"Input tensor: {name} | Received incompatible shape: {shape}.\nNote: Expected a shape compatible with: {meta.shape}"
            )
