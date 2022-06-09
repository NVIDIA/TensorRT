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
from tensorflow_quantization.custom_qdq_cases import EfficientNetQDQCase

from examples.data.data_loader import load_data
from examples.utils_finetuning import fine_tune

import numpy as np
import random
from utils import create_efficientnet_model

MODEL_VERSION = "b0"  # Options={b0, b3}

HYPERPARAMS = {
    # ################ Data loading ################
    "tfrecord_data_dir": "/media/Data/imagenet_data/tf_records",
    "batch_size": 64,
    "train_data_size": None,  # If 'None', consider all data, otherwise, consider subset.
    "val_data_size": None,  # If 'None', consider all data, otherwise, consider subset.
    # ############## Fine-tuning ##################
    "pretrained_ckpt_path": "./weights/efficientnet_{}/baseline".format(MODEL_VERSION),
    "epochs": 10,
    "steps_per_epoch": None,  # 'None' if you want to use the default number of steps. If you use this, make sure the number of steps is <= the number of shards (total number of samples / batch_size). Otherwise, an error will occur.
    "base_lr": 0.001,
    "optimizer": "piecewise_sgd",  # Options={sgd, piecewise_sgd, adam}
    "save_ckpt_dir": "./weights/efficientnet_{}/qat".format(MODEL_VERSION),
    # ############## Enable/disable tasks ##################
    "evaluate_baseline_model": True,
    "evaluate_qat_model": True,
    "seed": 42,
}

# Set seed for reproducible results
os.environ["PYTHONHASHSEED"] = str(HYPERPARAMS["seed"])
random.seed(HYPERPARAMS["seed"])
np.random.seed(HYPERPARAMS["seed"])
tf.random.set_seed(HYPERPARAMS["seed"])


def evaluate_acc(model, validation_data):
    # Compile model (needed to evaluate model)
    model.compile(
        optimizer="sgd",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )

    _, val_accuracy = model.evaluate(validation_data)

    return val_accuracy


def main():
    # ------------- Initial settings -------------
    # Load data
    train_batches, val_batches = load_data(HYPERPARAMS, model_name="efficientnet_"+MODEL_VERSION)

    # ------------- Baseline model -------------
    model = create_efficientnet_model(model_version=MODEL_VERSION)

    # Load pre-trained weights
    if HYPERPARAMS["pretrained_ckpt_path"]:
        model.load_weights(HYPERPARAMS["pretrained_ckpt_path"]).expect_partial()

    if HYPERPARAMS["evaluate_baseline_model"]:
        baseline_model_accuracy = evaluate_acc(model, val_batches)
        print("Baseline model accuracy:", baseline_model_accuracy)

    # ------------- QAT model -------------
    # Quantize model
    q_model = quantize_model(model, custom_qdq_cases=[EfficientNetQDQCase()])

    # Fine-tuning + saving new checkpoints
    print("Fine-tuning and saving QAT model checkpoint...")
    lr_schedule_array = [(1.0, 1), (0.1, 3), (0.01, 6), (0.001, 9), (0.001, 15)]
    fine_tune(
        q_model, train_batches, val_batches,
        qat_save_finetuned_weights=HYPERPARAMS["save_ckpt_dir"],
        hyperparams=HYPERPARAMS,
        lr_schedule_array=lr_schedule_array,
        enable_tensorboard_callback=False
    )
    print("Fine-tuning done!")

    # Loads best weights if they exist
    best_checkpoint_path = os.path.join(HYPERPARAMS["save_ckpt_dir"], "checkpoints_best")
    if os.path.exists(best_checkpoint_path + ".index"):
        q_model.load_weights(best_checkpoint_path).expect_partial()

    if HYPERPARAMS["evaluate_qat_model"]:
        print("\nEvaluating QAT model...")
        qat_model_accuracy = evaluate_acc(q_model, val_batches)
        print("QAT val accuracy:", qat_model_accuracy)


if __name__ == "__main__":
    main()
