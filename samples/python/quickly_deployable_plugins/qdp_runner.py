#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tensorrt as trt
import torch
import numpy as np

from polygraphy.backend.trt import (
    CreateConfig,
    TrtRunner,
    create_network,
    engine_from_network,
    network_from_onnx_path,
    bytes_from_engine,
    engine_from_bytes,
)

from polygraphy.backend.common import bytes_from_path
from polygraphy import cuda

import onnx_graphsurgeon as gs
import onnx
import os
import argparse

import tensorrt.plugin as trtp

import qdp_defs
import logging

def run_add(enable_autotune=False):

    if enable_autotune:
        qdp_defs.register_autotune()

    BLOCK_SIZE = 256

    builder, network = create_network(strongly_typed=True)
    x = torch.randint(10, (10, 3, 32, 32), dtype=torch.float32, device="cuda")

    # Populate network
    i_x = network.add_input(name="x", dtype=trt.DataType.FLOAT, shape=x.shape)

    out = network.add_plugin(
        trtp.op.sample.elemwise_add_plugin(i_x, block_size=BLOCK_SIZE)
    )
    out.get_output(0).name = "y"
    network.mark_output(tensor=out.get_output(0))

    builder.create_builder_config()

    engine = engine_from_network(
        (builder, network),
        CreateConfig(),
    )

    with TrtRunner(engine, "trt_runner") as runner:
        outputs = runner.infer(
            {
                "x": x,
            },
            copy_outputs_to_host=False,
        )

    if torch.allclose(x + 1, outputs["y"]):
        print("Inference result is correct!")
    else:
        print("Inference result is incorrect!")


def run_inplace_add():
    builder, network = create_network(strongly_typed=True)
    x = torch.ones((10, 3, 32, 32), dtype=torch.float32, device="cuda")

    x_clone = x.clone()

    i_x = network.add_input(name="x", dtype=trt.DataType.FLOAT, shape=x.shape)

    # Amounts to elementwise-add in the first and second plugins
    deltas = (2, 4)

    out0 = network.add_plugin(trtp.op.sample.elemwise_add_plugin_(i_x, delta=deltas[0]))
    out1 = network.add_plugin(
        trtp.op.sample.elemwise_add_plugin_(out0.get_output(0), delta=deltas[1])
    )
    out1.get_output(0).name = "y"
    network.mark_output(tensor=out1.get_output(0))

    builder.create_builder_config()

    # Enable preview feature for aliasing plugin I/O
    config = CreateConfig(
        preview_features=[trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03]
    )

    engine = engine_from_network(
        (builder, network),
        config,
    )

    context = engine.create_execution_context()

    stream = cuda.Stream()

    context.set_tensor_address("x", x.data_ptr())
    context.set_tensor_address("y", x.data_ptr())
    context.execute_async_v3(stream.ptr)
    stream.synchronize()

    if torch.allclose(x, x_clone + sum(deltas), atol=1e-2):
        print("Inference result is correct!")
    else:
        print("Inference result is incorrect!")
        print(x[0][0][0][:10])
        print(x_clone[0][0][0][:10])


def run_non_zero():
    builder, network = create_network(strongly_typed=True)
    inp_shape = (128, 128)

    X = np.random.normal(size=inp_shape).astype(trt.nptype(trt.DataType.FLOAT))

    # Zero out some random indices
    indices = np.random.choice(
        np.prod(inp_shape),
        replace=False,
        size=np.random.randint(0, np.prod(inp_shape) + 1),
    )
    X[np.unravel_index(indices, inp_shape)] = 0

    # Populate network
    i_x = network.add_input(name="X", dtype=trt.DataType.FLOAT, shape=inp_shape)

    out = network.add_plugin(trtp.op.sample.non_zero_plugin(i_x))
    out.get_output(0).name = "Y"
    network.mark_output(tensor=out.get_output(0))

    builder.create_builder_config()

    engine = engine_from_network(
        (builder, network),
        config=CreateConfig(),
    )

    Y_ref = np.transpose(np.nonzero(X))

    with TrtRunner(engine, "trt_runner") as runner:
        outputs = runner.infer({"X": X})
        Y = outputs["Y"]
        Y = Y[np.lexsort(np.fliplr(Y).T)]

    if np.allclose(Y, Y_ref, atol=1e-3):
        print("Inference result is correct!")
    else:
        print("Inference result is incorrect!")


def check_artifacts_dir_exists(artifacts_dir):
    if not os.path.exists(artifacts_dir):
        raise ValueError(f"artifacts_dir '{artifacts_dir}' does not exist")


def run_circ_pad(
    enable_multi_tactic=False, mode="onnx", artifacts_dir=None, save_or_load_engine=None, aot=False
):

    if enable_multi_tactic:
        qdp_defs.enable_multi_tactic_circ_pad()
    else:
        qdp_defs.enable_single_tactic_circ_pad()

    inp_shape = (10, 3, 32, 32)
    x = np.random.normal(size=inp_shape).astype(trt.nptype(trt.DataType.FLOAT))

    pads = np.array((1, 1, 1, 1), dtype=np.int32)

    if save_or_load_engine is not None and save_or_load_engine is False:
        check_artifacts_dir_exists(artifacts_dir)
        engine_path = os.path.join(artifacts_dir, "circ_pad.engine")
        engine = engine_from_bytes(bytes_from_path(engine_path))
    else:
        if mode == "inetdef":
            builder, network = create_network(strongly_typed=True)
            i_x = network.add_input(name="x", dtype=trt.DataType.FLOAT, shape=x.shape)
            out = network.add_plugin(trtp.op.sample.circ_pad_plugin(i_x, pads=pads), aot = aot)
            out.get_output(0).name = "y"
            network.mark_output(tensor=out.get_output(0))

            engine = engine_from_network(
                (builder, network),
                CreateConfig(),
            )
        elif mode == "onnx":
            if artifacts_dir is None:
                raise ValueError("'artifacts_dir' must be specified in onnx mode")

            check_artifacts_dir_exists(artifacts_dir)

            onnx_path = os.path.join(artifacts_dir, "circ_pad.onnx")
            var_x = gs.Variable(name="x", shape=inp_shape, dtype=np.float32)
            var_y = gs.Variable(name="y", dtype=np.float32)
            circ_pad_node = gs.Node(
                name="circ_pad_plugin 0",
                op="circ_pad_plugin",
                inputs=[var_x],
                outputs=[var_y],
                attrs={"pads": pads, "plugin_namespace": "sample", "aot": aot},
            )
            graph = gs.Graph(
                nodes=[circ_pad_node], inputs=[var_x], outputs=[var_y], opset=16
            )
            onnx.save(gs.export_onnx(graph), onnx_path)

            engine = engine_from_network(
                network_from_onnx_path(onnx_path, strongly_typed=True), CreateConfig()
            )
        else:
            raise ValueError(f"Unknown mode {mode}")

        if save_or_load_engine is not None and save_or_load_engine is True:
            check_artifacts_dir_exists(artifacts_dir)
            engine_path = os.path.join(artifacts_dir, "circ_pad.engine")
            with open(engine_path, "wb") as f:
                f.write(bytes_from_engine(engine))

    Y_ref = np.pad(x, [[0, 0], [0, 0], [pads[0], pads[1]], [pads[2], pads[3]]], "wrap")

    with TrtRunner(engine, "trt_runner") as runner:
        outputs = runner.infer({"x": x})
        Y = outputs["y"]

        if np.allclose(Y, Y_ref, atol=1e-2):
            print("Inference result is correct!")
        else:
            print("Inference result is incorrect!")


def setup_add_sample(subparsers):
    subparser = subparsers.add_parser("add", help="'add' sample help")
    subparser.add_argument("--autotune", action="store_true", help="Enable autotuning")
    subparser.add_argument("--aot", action="store_true", help="Use the AOT implementation of the plugin")
    subparser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable more verbose log output"
    )


def setup_inplace_add_sample(subparsers):
    subparser = subparsers.add_parser("inplace_add", help="inplace_add sample help")
    subparser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable more verbose log output"
    )


def setup_non_zero_sample(subparsers):
    subparser = subparsers.add_parser("non_zero", help="non_zero sample help")
    subparser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable more verbose log output"
    )


def setup_circ_pad_sample(subparsers):
    subparser = subparsers.add_parser("circ_pad", help="circ_pad sample help.")
    subparser.add_argument(
        "--multi_tactic", action="store_true", help="Enable multiple tactics."
    )
    subparser.add_argument(
        "--save_engine", action="store_true", help="Save engine to the artifacts_dir."
    )
    subparser.add_argument(
        "--load_engine",
        action="store_true",
        help="Load engine from the artifacts_dir. Ignores all other options.",
    )
    subparser.add_argument(
        "--artifacts_dir",
        type=str,
        help="Whether to store (or retrieve) artifacts.",
    )
    subparser.add_argument(
        "--mode",
        type=str,
        choices=["onnx", "inetdef"],
        help="Whether to use ONNX parser or INetworkDefinition APIs to construct the network.",
    )
    subparser.add_argument("--aot", action="store_true", help="Use the AOT implementation of the plugin.")
    subparser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose log output."
    )

    return subparser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="Main script help")
    subparsers = parser.add_subparsers(dest="sample", help="Mode help", required=True)

    setup_add_sample(subparsers)
    setup_inplace_add_sample(subparsers)
    circ_pad_subparser = setup_circ_pad_sample(subparsers)
    setup_non_zero_sample(subparsers)

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("QuicklyDeployablePlugins").setLevel(logging.DEBUG)

    if args.sample == "add":
        run_add(args.autotune)
    if args.sample == "inplace_add":
        run_inplace_add()
    if args.sample == "non_zero":
        run_non_zero()
    if args.sample == "circ_pad":
        if args.mode == "onnx":
            if args.artifacts_dir is None:
                parser.error(
                    "circ_pad: argument --mode: When mode is 'onnx', artifacts_dir is required"
                )

        save_or_load_engine = None

        if args.load_engine is True:
            if args.save_engine is True:
                parser.error(
                    "circ_pad: save_engine and load_engine cannot be specified at the same time. First save_engine and load_engine separately."
                )
            else:
                if args.multi_tactic is True or args.mode is not None:
                    print(
                        "warning circ_pad: when load_engine is specified, all other options except 'artifacts_dir' is ignored."
                    )

            save_or_load_engine = False
        else:
            if args.mode is None:
                circ_pad_subparser.print_help()
                parser.error(
                    "circ_pad: '--mode' option is required."
                )

        if args.save_engine is True:
            save_or_load_engine = True

        if args.multi_tactic and args.aot:
            parser.error(
                "circ_pad: '--aot' is not supported when '--multi_tactic' is specified."
            )

        run_circ_pad(args.multi_tactic, args.mode, args.artifacts_dir, save_or_load_engine, args.aot)
