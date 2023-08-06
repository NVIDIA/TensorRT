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

from Seq2Seq.onnxrt import Seq2SeqOnnxRT
from BART.BARTModelConfig import BARTModelTRTConfig
from BART.export import BARTModelClass

class BARTOnnxRT(Seq2SeqOnnxRT):
    def __init__(
        self,
        config_class=BARTModelTRTConfig,
        description="Runs OnnxRT results for BART model.",
        **kwargs
    ):
        super().__init__(config_class, description=description, model_classes=BARTModelClass, **kwargs)

# Entry point
RUN_CMD = BARTOnnxRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))