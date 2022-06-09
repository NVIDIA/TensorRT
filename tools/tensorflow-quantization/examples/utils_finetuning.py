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
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""
The only code snippet inherited from TensorFlow is the 'PiecewiseConstantDecayWithWarmup' class.
    See that class description for the exact modifications.
"""

import os
import tensorflow as tf
import numpy as np
from examples.data.data_loader import _NUM_IMAGES
from datetime import datetime
from examples.utils import ensure_dir
from typing import Dict, List
import logging


def get_finetuned_weights_dirname(hyperparams: Dict) -> str:
    """
    Generates the directory name to save all files relevant to the model's quantization.

    Args:
        hyperparams (Dict): dictionary with necessary fine-tuning hyper-parameters.

    Returns:
        full_dirpath (str): path to directory where the fine-tuned model, log files, ... will be saved.
    """
    dirname = (
        "qat_"
        + "ep"
        + str(hyperparams["epochs"])
        + "_steps"
        + str(hyperparams["steps_per_epoch"])
        + "_baselr"
        + str(hyperparams["base_lr"])
        + "_"
        + str(hyperparams["optimizer"])
        + "_bs"
        + str(hyperparams["batch_size"])
    )
    full_dirpath = os.path.join(hyperparams["save_root_dir"], dirname)
    return full_dirpath


def compile_model(model):
    model.compile(
        optimizer="sgd",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def fine_tune(
    q_model: tf.keras.Model,
    train_batches: tf.data.Dataset,
    val_batches: tf.data.Dataset,
    qat_save_finetuned_weights: str,
    hyperparams: Dict,
    logger: logging.RootLogger = None,
    lr_schedule_array: List[tuple] = [(1.0, 1), (0.1, 2), (0.01, 7)],
    enable_tensorboard_callback: bool = True
) -> None:
    """
    Helper function to fine-tune QAT model.

    Args:
        q_model (tf.keras.Model): Keras model.
        train_batches (tf.data.Dataset): train dataset split in batches.
        val_batches (tf.data.Dataset): validation dataset split in batches.
        qat_save_finetuned_weights (str): path to directory where the fine-tuned model, log files, ... will be saved.
        hyperparams (Dict): dictionary with necessary fine-tuning hyper-parameters.
        logger (logging.RootLogger): used to save logs.
        lr_schedule_array (List[tuple]): list of tuples in the format '(multiplier, epoch to start)'.
        enable_tensorboard_callback (bool): enables tensorboard callback if True.

    Returns:
        None

    Raises:
        ValueError: raised when the given optimizer is not supported.
    """

    if hyperparams["optimizer"] == "piecewise_sgd":
        lr_schedule = PiecewiseConstantDecayWithWarmup(
            batch_size=hyperparams["batch_size"],
            epoch_size=_NUM_IMAGES["train"],  # for tfrecord
            warmup_epochs=lr_schedule_array[0][1],
            boundaries=list(p[1] for p in lr_schedule_array[1:]),
            multipliers=list(p[0] for p in lr_schedule_array),
            compute_lr_on_cpu=True,
            base_lr=hyperparams["base_lr"],
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    elif hyperparams["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=hyperparams["base_lr"], momentum=0.0
        )
    elif hyperparams["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(hyperparams["base_lr"])
    else:
        raise ValueError("Optimizer `{}` is not supported. Please add support.".format(hyperparams["optimizer"]))

    q_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Initialize TensorBoard visualization
    callbacks = []
    if enable_tensorboard_callback:
        logdir_root = os.path.join(qat_save_finetuned_weights, "logs")
        ensure_dir(logdir_root)
        logdir = os.path.join(logdir_root, datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard_callback)
    # Initialize ModelCheckpoint callback
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(qat_save_finetuned_weights, "checkpoints_best"),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",  # Save ckpt with max 'val_accuracy' (best)
        save_best_only=True,
    )
    callbacks.append(ckpt_callback)

    history = q_model.fit(
        train_batches,
        validation_data=val_batches,
        batch_size=hyperparams["batch_size"],
        epochs=hyperparams["epochs"],
        steps_per_epoch=hyperparams["steps_per_epoch"],
        callbacks=callbacks,
        # verbose=2 if save_log is True else 1  # 0 = silent, 1 = progress bar, 2 = one line per epoch.
    )

    # Save fine-tuning history to logfile
    if logger:
        logger.info("------ Per epoch -------")
        for ep in history.epoch:
            log_str = "Epoch {ep}/{total_ep}".format(
                ep=ep + 1, total_ep=history.params["epochs"]
            )
            for metric_name, metric_value in history.history.items():
                log_str += " - " + metric_name + ": {}".format(metric_value[ep])
            logger.info(log_str)
        logger.info("------------------------")

        # Save fine-tuned checkpoints
        logger.info("Saving fine-tuned checkpoints")
    q_model.save_weights(os.path.join(qat_save_finetuned_weights, "checkpoints_last"))


class PiecewiseConstantDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    """
    Piecewise constant decay with warmup schedule.

    Original codebase: TensorFlow's "official.vision.image_classification.resnet.common"
    Original URL: https://github.com/tensorflow/models/blob/master/official/legacy/image_classification/resnet/common.py

    Modification: base learning rate `base_lr` given as parameter instead of a global constant.
        PREVIOUS: self.rescaled_lr = BASE_LEARNING_RATE * batch_size / base_lr_batch_size
        CURRENT:  self.rescaled_lr = base_lr
    """

    def __init__(
        self,
        batch_size,
        epoch_size,
        warmup_epochs,
        boundaries,
        multipliers,
        compute_lr_on_cpu=True,
        name=None,
        base_lr=0.1,
    ):
        super(PiecewiseConstantDecayWithWarmup, self).__init__()
        if len(boundaries) != len(multipliers) - 1:
            raise ValueError(
                "The length of boundaries must be 1 less than the "
                "length of multipliers"
            )

        steps_per_epoch = epoch_size // batch_size

        self.rescaled_lr = base_lr
        self.step_boundaries = [np.int64(steps_per_epoch * x) for x in boundaries]
        self.lr_values = [self.rescaled_lr * m for m in multipliers]
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.compute_lr_on_cpu = compute_lr_on_cpu
        self.name = name

        self.learning_rate_ops_cache = {}

    def __call__(self, step):
        if tf.executing_eagerly():
            return self._get_learning_rate(step)

        # In an eager function or graph, the current implementation of optimizer
        # repeatedly call and thus create ops for the learning rate schedule. To
        # avoid this, we cache the ops if not executing eagerly.
        graph = tf.compat.v1.get_default_graph()
        if graph not in self.learning_rate_ops_cache:
            if self.compute_lr_on_cpu:
                with tf.device("/device:CPU:0"):
                    self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
            else:
                self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
        return self.learning_rate_ops_cache[graph]

    def _get_learning_rate(self, step):
        """Compute learning rate at given step."""
        with tf.name_scope("PiecewiseConstantDecayWithWarmup"):

            def warmup_lr(step):
                return self.rescaled_lr * (
                    tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
                )

            def piecewise_lr(step):
                if step.dtype == tf.float32:
                    self.step_boundaries = [
                        np.float32(bound) for bound in self.step_boundaries
                    ]
                elif step.dtype == tf.int64:
                    self.step_boundaries = [
                        np.int64(bound) for bound in self.step_boundaries
                    ]

                step_lr = tf.compat.v1.train.piecewise_constant(
                    step, self.step_boundaries, self.lr_values
                )
                return step_lr

            return tf.cond(
                step < self.warmup_steps,
                lambda: warmup_lr(step),
                lambda: piecewise_lr(step),
            )

    def get_config(self):
        return {
            "rescaled_lr": self.rescaled_lr,
            "step_boundaries": self.step_boundaries,
            "lr_values": self.lr_values,
            "warmup_steps": self.warmup_steps,
            "compute_lr_on_cpu": self.compute_lr_on_cpu,
            "name": self.name,
        }
