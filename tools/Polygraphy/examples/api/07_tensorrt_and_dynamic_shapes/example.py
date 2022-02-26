#!/usr/bin/env python3
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
    # Note that we can freely inter-mix lazy and immediately-evaluated loaders.
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

    # NOTE: The following two lines will cause TensorRT to display errors since profile 0
    # is already in use by the first execution context. We'll suppress them using G_LOGGER.verbosity().
    #
    with G_LOGGER.verbosity(G_LOGGER.CRITICAL):
        dynamic_batching = TrtRunner(engine.create_execution_context())
        offline = TrtRunner(engine.create_execution_context())
        # NOTE: We could update the profile index here (e.g. `context.active_optimization_profile = 2`),
        # but instead, we'll use TrtRunner's `set_profile()` API when we later activate the runner.

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
            # NOTE: The very first time we activate this runner, we need to set
            # the profile index (it's 0 by default). We need to do this *only once*.
            # Alternatively, we could have set the profile index in the context directly (see above).
            #
            dynamic_batching.set_profile(1)  # Use the second profile, which is intended for dynamic batching.

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
        # NOTE: We must set the profile to something other than 0 or 1 since both of those
        # are now in use by the `low_latency` and `dynamic_batching` runners respectively.
        #
        offline.set_profile(2)  # Use the third profile, which is intended for the offline case.

        large_offline_batch = np.repeat(input_img, 128, axis=0)  # Shape: (128, 3, 28, 28)
        outputs = offline.infer({"X": large_offline_batch})
        assert np.array_equal(outputs["Y"], large_offline_batch)

        print("Offline runner succeeded!")


if __name__ == "__main__":
    main()
