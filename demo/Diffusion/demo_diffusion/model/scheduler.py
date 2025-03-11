#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from demo_diffusion.model import load


def make_scheduler(cls, version, pipeline, hf_token, framework_model_dir, subfolder="scheduler"):
    scheduler_dir = os.path.join(
        framework_model_dir, version, pipeline.name, next(iter({cls.__name__})).lower(), subfolder
    )
    if not os.path.exists(scheduler_dir):
        scheduler = cls.from_pretrained(load.get_path(version, pipeline), subfolder=subfolder, token=hf_token)
        scheduler.save_pretrained(scheduler_dir)
    else:
        print(f"[I] Load Scheduler {cls.__name__} from: {scheduler_dir}")
        scheduler = cls.from_pretrained(scheduler_dir)
    return scheduler
