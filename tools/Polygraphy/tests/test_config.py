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

import importlib
import os

from polygraphy import config


class TestConfigEnvVars:
    def test_ask_before_install(self):
        key = "POLYGRAPHY_ASK_BEFORE_INSTALL"
        prev = os.environ.pop(key, None)
        try:
            importlib.reload(config)
            assert not config.ASK_BEFORE_INSTALL

            os.environ[key] = "0"
            importlib.reload(config)
            assert not config.ASK_BEFORE_INSTALL

            os.environ[key] = "1"
            importlib.reload(config)
            assert config.ASK_BEFORE_INSTALL
        finally:
            if prev is not None:
                os.environ[key] = prev
            else:
                os.environ.pop(key, None)
            importlib.reload(config)
