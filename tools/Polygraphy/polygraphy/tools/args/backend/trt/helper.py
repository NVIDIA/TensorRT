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


# This file would have been called `util.py` but if we do that, then for some reason Python thinks
# that this is the file we want when importing `polygraphy.tools.args.util`.
from polygraphy.tools.script import inline, inline_identifier, safe


def make_trt_enum_val(enum_name, value):
    """
    Helper function to create inline TRT enums for usage across various TRT classes.
    """
    return inline(safe("trt.{:}.{:}", inline_identifier(enum_name), inline_identifier(value.upper())))
