#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from polygraphy import util
from polygraphy.logger.logger import Logger


# We don't use the global logger here because we would have to reset the state each time.
class TestLogger(object):
    def test_log_file(self):
        logger = Logger()
        with util.NamedTemporaryFile("w+") as log_file:
            logger.log_file = log_file.name
            assert logger.log_file == log_file.name
            logger.info("Hello")

            log_file.seek(0)
            assert log_file.read() == "[I] Hello\n"
