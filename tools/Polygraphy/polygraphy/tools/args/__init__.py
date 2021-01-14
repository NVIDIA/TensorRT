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
from polygraphy.tools.args.data_loader import DataLoaderArgs
from polygraphy.tools.args.logger import LoggerArgs
from polygraphy.tools.args.model import ModelArgs
from polygraphy.tools.args.onnx.loader import OnnxLoaderArgs
from polygraphy.tools.args.onnx.runner import OnnxtfRunnerArgs
from polygraphy.tools.args.onnxrt.runner import OnnxrtRunnerArgs
from polygraphy.tools.args.tf2onnx.loader import Tf2OnnxLoaderArgs
from polygraphy.tools.args.tf.config import TfConfigArgs
from polygraphy.tools.args.tf.loader import TfLoaderArgs
from polygraphy.tools.args.tf.runner import TfRunnerArgs
from polygraphy.tools.args.trt.loader import TrtLoaderArgs
from polygraphy.tools.args.trt.runner import TrtRunnerArgs
from polygraphy.tools.args.trt_legacy import TrtLegacyArgs
from polygraphy.tools.args.comparator import ComparatorRunArgs, ComparatorCompareArgs
