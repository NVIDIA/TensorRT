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
import sys

# This sample uses an MNIST PyTorch model to create a TensorRT Inference Engine
import model
import numpy as np
import pycuda.autoinit
import tensorrt as trt

sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))

import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


# Populate the TRT network, injecting some dummy weights
def populate_network_with_some_dummy_weights(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # Set dummy weights for the kernel and bias weights in the conv1 layer. We
    # will refit the engine with the actual weights later.
    conv1_w = np.zeros((20, 5, 5), dtype=np.float32)
    conv1_b = np.zeros(20, dtype=np.float32)

    conv1 = network.add_convolution(
        input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b
    )
    conv1.name = "conv_1"
    conv1.stride = (1, 1)
    # Associate weights with name and refit weights via name later in refitter.
    network.set_weights_name(conv1_w, "conv1.weight")

    pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)

    conv2_w = weights["conv2.weight"].numpy()
    conv2_b = weights["conv2.bias"].numpy()
    conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
    conv2.stride = (1, 1)

    pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
    pool2.stride = (2, 2)

    def add_matmul_as_fc(net, input, outputs, w, b):
        assert len(input.shape) >= 3
        m = 1 if len(input.shape) == 3 else input.shape[0]
        k = int(np.prod(input.shape) / m)
        assert np.prod(input.shape) == m * k
        n = int(w.size / k)
        assert w.size == n * k
        assert b.size == n

        input_reshape = net.add_shuffle(input)
        input_reshape.reshape_dims = trt.Dims2(m, k)

        filter_const = net.add_constant(trt.Dims2(n, k), w)
        mm = net.add_matrix_multiply(
            input_reshape.get_output(0),
            trt.MatrixOperation.NONE,
            filter_const.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        )

        bias_const = net.add_constant(trt.Dims2(1, n), b)
        bias_add = net.add_elementwise(mm.get_output(0), bias_const.get_output(0), trt.ElementWiseOperation.SUM)

        output_reshape = net.add_shuffle(bias_add.get_output(0))
        output_reshape.reshape_dims = trt.Dims4(m, n, 1, 1)
        return output_reshape

    fc1_w = weights["fc1.weight"].numpy()
    fc1_b = weights["fc1.bias"].numpy()
    fc1 = add_matmul_as_fc(network, pool2.get_output(0), 500, fc1_w, fc1_b)

    relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)

    fc2_w = weights["fc2.weight"].numpy()
    fc2_b = weights["fc2.bias"].numpy()
    fc2 = add_matmul_as_fc(network, relu1.get_output(0), ModelData.OUTPUT_SIZE, fc2_w, fc2_b)

    fc2.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=fc2.get_output(0))


# Build a TRT engine, but leave out some weights
def build_engine_with_some_missing_weights(weights):
    # For more information on TRT basics, refer to the introductory samples.
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    config = builder.create_builder_config()
    runtime = trt.Runtime(TRT_LOGGER)

    config.max_workspace_size = common.GiB(1)
    # Set the refit flag in the builder
    config.set_flag(trt.BuilderFlag.REFIT)
    # Populate the network using weights from the PyTorch model.
    populate_network_with_some_dummy_weights(network, weights)
    # Build and return an engine.
    plan = builder.build_serialized_network(network, config)
    return runtime.deserialize_cuda_engine(plan)


# Copy an image to the pagelocked input buffer
def load_img_to_input_buffer(img, pagelocked_buffer):
    np.copyto(pagelocked_buffer, img)


# Get the accuracy on the test set using TensorRT
def get_trt_test_accuracy(engine, inputs, outputs, bindings, stream, mnist_model):
    context = engine.create_execution_context()
    correct = 0
    total = 0
    # Run inference on every sample.
    # Technically this could be batched, however this only comprises a fraction of total
    # time spent in the test.
    for test_img, test_name in mnist_model.get_all_test_samples():
        load_img_to_input_buffer(test_img, pagelocked_buffer=inputs[0].host)
        # For more information on performing inference, refer to the introductory samples.
        # The common.do_inference function will return a list of outputs - we only have one in this case.
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        pred = np.argmax(output)
        correct += test_name == pred
        total += 1

    accuracy = float(correct) / total
    print("Got {} correct predictions out of {} ({:.1f}%)".format(correct, total, 100 * accuracy))

    return accuracy


def main():
    common.add_help(description="Runs an MNIST network using a PyTorch model")
    # Train the PyTorch model
    mnist_model = model.MnistModel()
    mnist_model.learn()
    weights = mnist_model.get_weights()
    # Do inference with TensorRT.
    engine = build_engine_with_some_missing_weights(weights)
    # Build an engine, allocate buffers and create a stream.
    # For more information on buffer allocation, refer to the introductory samples.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    print("Accuracy Before Engine Refit (random weights, expecting low accuracy)")
    get_trt_test_accuracy(engine, inputs, outputs, bindings, stream, mnist_model)

    # Refit the engine with the actual trained weights for the conv_1 layer.
    refitter = trt.Refitter(engine, TRT_LOGGER)
    # Set max threads that can be used by refitter.
    refitter.max_threads = 10

    # To get a list of all refittable layers and associated weightRoles
    # in the network, use refitter.get_all()
    # Set the actual weights for the conv_1 layer. Since it consists of
    # kernel weights and bias weights, set each of them by specifying
    # the WeightsRole.
    # Prefer to refit named weights via set_named_weights
    refitter.set_named_weights("conv1.weight", weights["conv1.weight"].numpy())
    # set_named_weights is not available for unnamed weights. Call set_weights instead.
    refitter.set_weights("conv_1", trt.WeightsRole.BIAS, weights["conv1.bias"].numpy())
    # Get missing weights names. This should return empty
    # lists in this case.
    missing_weights = refitter.get_missing_weights()
    assert (
        len(missing_weights) == 0
    ), "Refitter found missing weights. Call set_named_weights() or set_weights() for all missing weights"
    # Refit the engine with the new weights. This will return True if
    # the refit operation succeeded.
    assert refitter.refit_cuda_engine()

    expected_correct_predictions = mnist_model.get_latest_test_set_accuracy()
    print(
        "Accuracy After Engine Refit (expecting {:.1f}% correct predictions)".format(100 * expected_correct_predictions)
    )
    assert get_trt_test_accuracy(engine, inputs, outputs, bindings, stream, mnist_model) >= expected_correct_predictions


if __name__ == "__main__":
    main()
