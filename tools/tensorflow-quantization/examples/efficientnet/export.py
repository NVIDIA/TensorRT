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


import tensorflow as tf
import argparse
from utils import create_efficientnet_model
from tensorflow_quantization.quantize import quantize_model
from tensorflow_quantization.custom_qdq_cases import EfficientNetQDQCase


def export_saved_model(model_version="b0"):
    model = create_efficientnet_model(model_version=model_version)
    q_model = quantize_model(model, custom_qdq_cases=[EfficientNetQDQCase()])
    if args.ckpt:
        q_model.load_weights(args.ckpt).expect_partial()

    tf.keras.models.save_model(q_model, args.output)
    print("Exported the model to {}".format(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export saved model for efficientnet_b0"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="qat/checkpoints_best",
        help="Path to pretrained QAT efficientnet checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qat/saved_model",
        help="Path to pretrained QAT saved model.",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="b0",
        help="EfficientNet model version, currently supports {'b0', 'b3'}.",
    )
    args = parser.parse_args()
    export_saved_model(args.model_version)
