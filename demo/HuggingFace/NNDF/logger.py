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

import logging

G_LOGGER = logging.getLogger("OSS")
G_LOGGER.DEBUG = logging.DEBUG
G_LOGGER.INFO = logging.INFO
G_LOGGER.WARNING = logging.WARNING
G_LOGGER.ERROR = logging.ERROR

formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
stream = logging.StreamHandler()
stream.setFormatter(formatter)
G_LOGGER.addHandler(stream)
