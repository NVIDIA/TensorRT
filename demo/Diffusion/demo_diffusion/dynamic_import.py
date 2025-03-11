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

import warnings
from importlib import import_module


def import_from_diffusers(model_name, module_name):
    try:
        module = import_module(module_name)
        return getattr(module, model_name)
    except ImportError:
        warnings.warn(f"Failed to import {module_name}. The {model_name} model will not be available.", ImportWarning)
    except AttributeError:
        warnings.warn(f"The {model_name} model is not available in the installed version of diffusers.", ImportWarning)
    return None
