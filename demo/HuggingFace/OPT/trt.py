#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

from Seq2Seq.trt import Seq2SeqTRT
from OPT.OPTModelConfig import OPTModelTRTConfig
from OPT.export import OPTModelClass


class OPTTRT(Seq2SeqTRT):
    def __init__(
        self,
        config_class=OPTModelTRTConfig,
        description="Runs trt results for OPT model.",
        **kwargs
    ):
        super().__init__(config_class, description=description, model_classes=OPTModelClass, **kwargs)

# Entry point
RUN_CMD = OPTTRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))