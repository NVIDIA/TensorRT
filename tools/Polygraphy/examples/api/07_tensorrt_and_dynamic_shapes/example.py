#!/usr/bin/env python3
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

"""
This script builds an engine with 3 separate optimization profiles, each
built for a specific use-case. It then creates 3 separate execution contexts
and corresponding `TrtRunner`s for inference.
"""
import numpy as np
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER


def main():
    # A Profile maps each input tensor to a range of shapes.
    # The `add()` method can be used to add shapes for a single input.
    #
    # TIP: To save lines, calls to `add` can be chained:
    #     profile.add("input0", ...).add("input1", ...)
    #
    #   Of course, you may alternatively write this as:
    #     profile.add("input0", ...)
    #     profile.add("input1", ...)
    #
    profiles = [
        # The low-latency case. For best performance, min == opt == max.
        Profile().add("X", min=(1, 3, 28, 28), opt=(1, 3, 28, 28), max=(1, 3, 28, 28)),
        # The dynamic batching case. We use `4` for the opt batch size since that's our most common case.
        Profile().add("X", min=(1, 3, 28, 28), opt=(4, 3, 28, 28), max=(32, 3, 28, 28)),
        # The offline case. For best performance, min == opt == max.
        Profile().add("X", min=(128, 3, 28, 28), opt=(128, 3, 28, 28), max=(128, 3, 28, 28)),
    ]

    # See examples/api/06_immediate_eval_api for details on immediately evaluated functional loaders like `engine_from_network`.
    # Note that we can freely mix lazy and immediately-evaluated loaders.
    engine = engine_from_network(
        network_from_onnx_path("dynamic_identity.onnx"), config=CreateConfig(profiles=profiles)
    )

    # We'll save the engine so that we can inspect it with `inspect model`.
    # This should make it easy to see how the engine bindings are laid out.
    save_engine(engine, "dynamic_identity.engine")

    # We'll create, but not activate, three separate runners, each with a separate context.
    #
    # TIP: By providing a context directly, as opposed to via a lazy loader,
    # we can ensure that the runner will *not* take ownership of it.
    #
    low_latency = TrtRunner(engine.create_execution_context())

    # NOTE: The following two lines may cause TensorRT to display errors since profile 0
    # is already in use by the first execution context. We'll suppress them using G_LOGGER.verbosity().
    #
    with G_LOGGER.verbosity(G_LOGGER.CRITICAL):
        # We can use the `optimization_profile` parameter of the runner to ensure that the correct optimization profile is used.
        # This eliminates the need to call `set_profile()` later.
        dynamic_batching = TrtRunner(
            engine.create_execution_context(), optimization_profile=1
        )  # Use the second profile, which is intended for dynamic batching.

        # For the sake of example, we *won't* use `optimization_profile` here.
        # Instead, we'll use `set_profile()` after activating the runner.
        offline = TrtRunner(engine.create_execution_context())

    # Finally, we can activate the runners as we need them.
    #
    # NOTE: Since the context and engine are already created, the runner will only need to
    # allocate input and output buffers during activation.

    input_img = np.ones((1, 3, 28, 28), dtype=np.float32)  # An input "image"

    with low_latency:
        outputs = low_latency.infer({"X": input_img})
        assert np.array_equal(outputs["Y"], input_img)  # It's an identity model!

        print("Low latency runner succeeded!")

        # While we're serving requests online, we might decide that we need dynamic batching
        # for a moment.
        #
        # NOTE: We're assuming that activating runners will be cheap here, so we can bring up
        # the dynamic batching runner just-in-time.
        #
        # TIP: If activating the runner is not cheap (e.g. input/output buffers are large),
        # it might be better to keep the runner active the whole time.
        #
        with dynamic_batching:
            # We'll create fake batches by repeating our fake input image.
            small_input_batch = np.repeat(input_img, 4, axis=0)  # Shape: (4, 3, 28, 28)
            outputs = dynamic_batching.infer({"X": small_input_batch})
            assert np.array_equal(outputs["Y"], small_input_batch)

    # If we need dynamic batching again later, we can activate the runner once more.
    #
    # NOTE: This time, we do *not* need to set the profile.
    #
    with dynamic_batching:
        # NOTE: We can use any shape that's in the range of the profile without
        # additional setup - Polygraphy handles the details behind the scenes!
        #
        large_input_batch = np.repeat(input_img, 16, axis=0)  # Shape: (16, 3, 28, 28)
        outputs = dynamic_batching.infer({"X": large_input_batch})
        assert np.array_equal(outputs["Y"], large_input_batch)

        print("Dynamic batching runner succeeded!")

    with offline:
        # NOTE: When we first activate this runner, we need to set the profile index (it's 0 by default).
        # Since we provided our own execution context when we created the runner, we need to do this *only once*.
        # Our settings persist since the context will remain alive even after the runner is deactivated.
        # If we had instead allowed the runner to own the context, we'd need to repeat this step each time we activated the runner.
        #
        # Alternatively, we could have used the `optimization_profile` parameter (see above).
        #
        offline.set_profile(2)  # Use the third profile, which is intended for the offline case.

        large_offline_batch = np.repeat(input_img, 128, axis=0)  # Shape: (128, 3, 28, 28)
        outputs = offline.infer({"X": large_offline_batch})
        assert np.array_equal(outputs["Y"], large_offline_batch)

        print("Offline runner succeeded!")


if __name__ == "__main__":
    main()
