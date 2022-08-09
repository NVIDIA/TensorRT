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


import os
import tensorflow as tf

from tensorflow_quantization.quantize import quantize_model
from tensorflow_quantization.utils import convert_saved_model_to_onnx
from tensorflow_quantization.custom_qdq_cases import InceptionQDQCase

from examples.utils import ensure_dir, get_tfkeras_model
from examples.data.data_loader import load_data
from examples.utils_finetuning import (
    get_finetuned_weights_dirname,
    fine_tune,
    compile_model,
)
import gc
import numpy as np
import random
import sys
import logging


MODEL_NAME = "inception_v3"  # Options=[inception_v3]

HYPERPARAMS = {
    # ################ Data loading ################
    "tfrecord_data_dir": "/media/Data/imagenet_data/tf_records",
    "batch_size": 64,
    "train_data_size": None,  # Only for `tfrecord`. If None, consider all data, otherwise, consider subset.
    "val_data_size": None,  # Only for `tfrecord`. If None, consider all data, otherwise, consider subset.
    # ############## Fine-tuning ##################
    "epochs": 10,
    "steps_per_epoch": 500,  # 500,  # 'None' if you want to use the default number of steps. If you use this, make sure the number of steps is <= the number of shards (total number of samples / batch_size). Otherwise, an error will occur.
    "base_lr": 0.001,  # 0.0001
    "optimizer": "piecewise_sgd",  # Options={sgd, piecewise_sgd, adam}
    "save_root_dir": "./weights/{}".format(
        MODEL_NAME
    ),  # DIR is updated to reflect hyperparams
    # ############## Enable/disable tasks ##################
    "finetune_qat_model": True,  # If True, finetune QAT model. Otherwise, just quantize and load weights if existent.
    "rewrite_weights_qat_finetuning": True,  # If True, rewrites existing fine-tuned weights. Otherwise, just load weights if they exist.
    "evaluate_baseline_model": True,
    "evaluate_qat_model": True,
    "save_baseline_model": True,
    "seed": 42,
}

# Set seed for reproducible results
os.environ["PYTHONHASHSEED"] = str(HYPERPARAMS["seed"])
random.seed(HYPERPARAMS["seed"])
np.random.seed(HYPERPARAMS["seed"])
tf.random.set_seed(HYPERPARAMS["seed"])

# Create logger and save to out.log
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


def main():
    # ------------- Initial settings -------------
    # Create directory to save the fine-tuned weights + add relevant hyperparameters in the name
    qat_save_finetuned_weights = get_finetuned_weights_dirname(HYPERPARAMS)
    ensure_dir(qat_save_finetuned_weights)

    # Add terminal and file handlers to logger
    output_file_handler = logging.FileHandler(
        os.path.join(qat_save_finetuned_weights, "out.log"), mode="w"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    LOGGER.addHandler(output_file_handler)
    LOGGER.addHandler(stdout_handler)

    # Load data
    train_batches, val_batches = load_data(HYPERPARAMS, model_name=MODEL_NAME)

    # ------------- Baseline model -------------
    LOGGER.info("------------- Baseline model -------------")

    # Instantiate Baseline model
    model = get_tfkeras_model(model_name=MODEL_NAME)

    if HYPERPARAMS["evaluate_baseline_model"]:
        # Compile model (needed to evaluate model)
        compile_model(model)
        _, baseline_model_accuracy = model.evaluate(val_batches)
        LOGGER.info("Baseline val accuracy: {}".format(baseline_model_accuracy))

    if HYPERPARAMS["save_baseline_model"]:
        tf.keras.models.save_model(
            model, os.path.join(HYPERPARAMS["save_root_dir"], "saved_model_baseline")
        )
        convert_saved_model_to_onnx(
            saved_model_dir=os.path.join(
                HYPERPARAMS["save_root_dir"], "saved_model_baseline"
            ),
            onnx_model_path=os.path.join(
                HYPERPARAMS["save_root_dir"], "model_baseline.onnx"
            ),
        )

    # ------------- QAT model -------------
    # Quantize model
    LOGGER.info("\n------------- QAT model -------------")
    q_model = quantize_model(model, custom_qdq_cases=[InceptionQDQCase()])
    # q_model = quantize_model(model)

    finetuned_qat_weights_path = os.path.join(
        qat_save_finetuned_weights, "checkpoints_best"
    )
    # Performs fine-tuning if `rewrite` is enabled or if fine-tuned weights don't exist yet
    #   (1st time fine-tuning model).
    if HYPERPARAMS["finetune_qat_model"] and (
        HYPERPARAMS["rewrite_weights_qat_finetuning"]
        or not os.path.exists(finetuned_qat_weights_path)
    ):
        # Fine-tuning + saving new checkpoints
        LOGGER.info("\nFine-tuning model...")
        fine_tune(
            q_model,
            train_batches,
            val_batches,
            qat_save_finetuned_weights,
            HYPERPARAMS,
            LOGGER,
        )
        LOGGER.info("Fine-tuning done!")

    # Loads best weights if they exist
    if os.path.exists(finetuned_qat_weights_path + ".index"):
        LOGGER.info("Loading fine-tuned weights...")
        q_model.load_weights(finetuned_qat_weights_path).expect_partial()
        LOGGER.info("Loaded complete!")
    compile_model(q_model)

    if HYPERPARAMS["evaluate_qat_model"]:
        LOGGER.info("\nEvaluating QAT model...")
        _, qat_model_accuracy = q_model.evaluate(val_batches)
        LOGGER.info("QAT val accuracy: {}".format(qat_model_accuracy))

    # Save quantized model
    LOGGER.info("\nSaving QAT model")
    tf.keras.models.save_model(
        q_model, os.path.join(qat_save_finetuned_weights, "saved_model")
    )

    # Clear GPU and invoke Garbage Collector to avoid script ending during ONNX conversion
    tf.keras.backend.clear_session()
    gc.collect()
    del model
    del q_model

    # Convert SavedModel to ONNX
    LOGGER.info("\nONNX conversion...")
    convert_saved_model_to_onnx(
        saved_model_dir=os.path.join(qat_save_finetuned_weights, "saved_model"),
        onnx_model_path=os.path.join(qat_save_finetuned_weights, "model.onnx"),
    )


if __name__ == "__main__":
    main()
