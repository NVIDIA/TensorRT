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

# This file contains functions for training a TensorFlow model
import tensorflow as tf
import tensorrt as trt
import numpy as np
import os

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))

MODEL_DIR = os.path.join(WORKING_DIR, "models")

# MNIST dataset metadata
MNIST_IMAGE_SIZE = 28
MNIST_CHANNELS = 1
MNIST_CLASSES = 10


class ModelData(object):
    INPUT_NAME = "InputLayer"
    INPUT_SHAPE = (MNIST_CHANNELS, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
    RELU6_NAME = "ReLU6"
    OUTPUT_NAME = "OutputLayer/Softmax"
    OUTPUT_SHAPE = (MNIST_IMAGE_SIZE,)
    DATA_TYPE = trt.float32


def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.reshape(x_train, (-1, 1, 28, 28))
    x_test = np.reshape(x_test, (-1, 1, 28, 28))
    return x_train, y_train, x_test, y_test


def build_model():
    # Create the keras model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[1, 28, 28], name="InputLayer"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation(activation=tf.nn.relu6, name="ReLU6"))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="OutputLayer"))
    return model


def train_model():
    # Build and compile model
    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load data
    x_train, y_train, x_test, y_test = load_data()
    np.save(os.path.join(MODEL_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)

    # Train the model on the data
    model.fit(x_train, y_train, epochs=10, verbose=1)

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test loss: {}\nTest accuracy: {}".format(test_loss, test_acc))

    return model


def maybe_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_model(model):
    output_names = model.output.op.name
    sess = tf.keras.backend.get_session()

    graphdef = sess.graph.as_graph_def()
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, [output_names])
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    # Make directory to save model in if it doesn't exist already
    maybe_mkdir(MODEL_DIR)

    model_path = os.path.join(MODEL_DIR, "trained_lenet5.pb")
    with open(model_path, "wb") as ofile:
        ofile.write(frozen_graph.SerializeToString())


if __name__ == "__main__":
    model = train_model()
    save_model(model)
