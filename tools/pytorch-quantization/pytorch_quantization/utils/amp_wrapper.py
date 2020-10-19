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


"""This makes the amp decorators work as normal when amp is available, and otherwise act as no-op pass-throughs."""

from absl import logging

try:
    import apex.amp as amp
    def half_function(fn):
        return amp.half_function(fn)
    def float_function(fn):
        return amp.float_function(fn)
    def promote_function(fn):
        return amp.promote_function(fn)
except Exception:
    logging.error("AMP is not avaialble.")
    def half_function(fn):
        return fn
    def float_function(fn):
        return fn
    def promote_function(fn):
        return fn
