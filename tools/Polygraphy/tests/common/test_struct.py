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
import numpy as np
from polygraphy.common import TensorMetadata


class TestTensorMetadata:
    def test_str(self):
        meta = TensorMetadata().add("X", dtype=np.float32, shape=(64, 64))
        assert str(meta) == "{X [dtype=float32, shape=(64, 64)]}"

    def test_str_no_dtype(self):
        meta = TensorMetadata().add("X", dtype=None, shape=(64, 64))
        assert str(meta) == "{X [shape=(64, 64)]}"

    def test_str_no_shape(self):
        meta = TensorMetadata().add("X", dtype=np.float32, shape=None)
        assert str(meta) == "{X [dtype=float32]}"

    def test_str_no_meta(self):
        meta = TensorMetadata().add("X", dtype=None, shape=None)
        assert str(meta) == "{X}"
