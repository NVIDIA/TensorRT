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
import contextlib
from collections import OrderedDict

from polygraphy import cuda, mod, util
from polygraphy.exception import PolygraphyException
from polygraphy.logger import G_LOGGER, LogMode

trt = mod.lazy_import("tensorrt")
np = mod.lazy_import("numpy")


@mod.export()
def Calibrator(
    data_loader, cache=None, BaseClass=None, batch_size=None, quantile=None, regression_cutoff=None, algo=None
):
    """
    Supplies calibration data to TensorRT to calibrate the network for INT8 inference.

    Args:
        data_loader (Sequence[OrderedDict[str, Union[numpy.ndarray, DeviceView, int]]]):
            A generator or iterable that yields a dictionary that maps input names to NumPy
            arrays, Polygraphy DeviceViews, or GPU pointers. If NumPy arrays or DeviceViews
            are provided, the calibrator will check the data types and shapes if possible
            to ensure that they match those expected by the model.

            In case you don't know details about the inputs ahead of time, you can access the
            `input_metadata` property in your data loader, which will be set to a ``TensorMetadata``
            instance by Polygraphy APIs like ``CreateConfig`` and ``EngineFromNetwork``.
            Note that this does not work for generators or lists.

            The number of calibration batches is controlled by the number of items supplied
            by the data loader.


        cache (Union[str, file-like]):
                Path or file-like object to save/load the calibration cache.
                By default, the calibration cache is not saved.
        BaseClass (type):
                The type of calibrator to inherit from.
                Defaults to ``trt.IInt8EntropyCalibrator2``.
        batch_size (int):
                [DEPRECATED] The size of each batch provided by the data loader.
        quantile (float):
                The quantile to use for ``trt.IInt8LegacyCalibrator``.
                Has no effect for other calibrator types.
                Defaults to 0.5.
        regression_cutoff (float):
                The regression cutoff to use for ``trt.IInt8LegacyCalibrator``.
                Has no effect for other calibrator types.
                Defaults to 0.5.
        algo (trt.CalibrationAlgoType):
                Calibration algorithm to use for ``trt.IInt8Calibrator``.
                Has no effect for other calibrator types.
                Defaults to ``trt.CalibrationAlgoType.MINMAX_CALIBRATION``.
    """
    BaseClass = util.default(BaseClass, trt.IInt8EntropyCalibrator2)

    class CalibratorClass(BaseClass):
        """
        Calibrator that supplies calibration data to TensorRT to calibrate the network for INT8 inference.
        """

        def __init__(self):
            # Must explicitly initialize parent for any trampoline class! Will mysteriously segfault without this.
            BaseClass.__init__(self)

            self.data_loader = data_loader
            self._cache = cache
            self.device_buffers = OrderedDict()
            self.input_metadata = None
            self.reset()
            G_LOGGER.verbose(f"Created calibrator [cache={self._cache}]")

            self.batch_size = util.default(batch_size, 1)

            self.is_polygraphy_calibrator = True
            # The function that constructed this instance
            self.make_func = Calibrator

        def set_input_metadata(self, input_metadata):
            """
            Sets the input metadata for the calibrator.

            This is passed along to the data loader and is also used for
            input data type and shape checks.

            NOTE: This generally does not need to be called manually if the calibrator is being used
            with Polygraphy's loaders, like ``CreateConfig`` or ``EngineFromNetwork``.

            Args:
                input_metadata (TensorMetadata):
                        Mapping of input names to their data types and shapes.
                        Passed along to the data loader if provided. This is required if
                        using Polygraphy's included `DataLoader` to provide calibration data,
                        or if data type and shape checking is desired.
            """
            self.input_metadata = input_metadata
            if input_metadata is not None:
                with contextlib.suppress(AttributeError):
                    self.data_loader.input_metadata = input_metadata

        def reset(self, input_metadata=None):
            """
            Reset this calibrator for reuse.

            The calibrator will clear any dynamic ranges cached from previous calibration runs, and will
            attempt to rewind the data loader (note that generators cannot be rewound).

            Typically, this is only required if the same calibrator is used for multiple different networks.
            """
            # Attempt to reset data loader
            self.data_loader_iter = iter(self.data_loader)
            self.num_batches = 0

            # Make sure calibrator will check the cache again when reset.
            self.cache_contents = None

            if input_metadata is not None:
                mod.warn_deprecated("input_metadata parameter", "set_input_metadata", remove_in="0.45.0")
                self.set_input_metadata(input_metadata)

        def get_batch_size(self):
            return self.batch_size

        def _get_batch_impl(self, names):
            try:
                buffers = next(self.data_loader_iter)
            except StopIteration:
                if not self.num_batches:
                    G_LOGGER.critical(
                        "Calibrator data loader provided no data.\nPossible reasons for this include:\n(1) data loader "
                        "has no data to provide\n(2) data loader was a generator, and the calibrator is being "
                        "used multiple times (generators cannot be rewound)"
                    )
                return None
            else:
                self.num_batches += 1

            util.check_sequence_contains(
                buffers.keys(),
                names,
                name="calibration input data provided by the data loader",
                items_name="inputs",
            )

            def check_buffer(name, buffer):
                if self.input_metadata is None:
                    return

                expected_dtype, expected_shape = self.input_metadata[name]

                err_prefix = "Received an unexpected input from the data loader during calibration. "
                if buffer.dtype != expected_dtype:
                    G_LOGGER.critical(
                        err_prefix
                        + f"For input: '{name}', expected data type: {expected_dtype}, but received: {buffer.dtype}"
                    )

                if not util.is_valid_shape_override(buffer.shape, expected_shape):
                    G_LOGGER.critical(
                        err_prefix
                        + f"For input: '{name}', expected a shape compatible with: {expected_shape}, but received: {buffer.shape}"
                    )

            ptrs = []
            for name in names:
                buf = buffers[name]

                if isinstance(buf, cuda.DeviceView):
                    check_buffer(name, buf)
                    ptrs.append(buf.ptr)
                elif isinstance(buf, np.ndarray):
                    check_buffer(name, buf)
                    if name not in self.device_buffers:
                        self.device_buffers[name] = cuda.DeviceArray(shape=buf.shape, dtype=buf.dtype)
                        G_LOGGER.verbose(f"Allocated: {self.device_buffers[name]}")

                    self.device_buffers[name].resize(buf.shape)
                    buf = util.make_contiguous(buf)
                    ptrs.append(self.device_buffers[name].copy_from(buf).ptr)
                elif isinstance(buf, int):
                    ptrs.append(buf)
                else:
                    G_LOGGER.critical(
                        f"Calibration data loader provided an unrecognized type: {type(buf).__name__} for input: {name}."
                        "\nPlease provide either a NumPy array, Polygraphy DeviceView, or GPU pointer. "
                    )

            return ptrs

        def get_batch(self, names):
            ptrs = None
            try:
                ptrs = self._get_batch_impl(names)
            except PolygraphyException:
                pass
            if ptrs is None:
                self.free()
            return ptrs

        def read_calibration_cache(self):
            def load_from_cache():
                if self._cache is None or not util.get_file_size(self._cache):
                    return None

                try:
                    return util.load_file(self._cache, description="calibration cache")
                except Exception as err:
                    G_LOGGER.error(f"Could not read from calibration cache: {self._cache}\nNote: Error was: {err}")
                    return None

            if self.cache_contents is not None:
                return self.cache_contents

            self.cache_contents = load_from_cache()

            if not self.cache_contents:
                if self.cache_contents is not None:
                    G_LOGGER.warning(
                        "Calibration cache was provided, but is empty. "
                        "Will regenerate scales by running calibration.",
                        mode=LogMode.ONCE,
                    )
                self.cache_contents = None

            return self.cache_contents

        def write_calibration_cache(self, cache):
            self.cache_contents = cache.tobytes()

            if self._cache is None:
                return

            try:
                util.save_file(contents=self.cache_contents, dest=self._cache, description="calibration cache")
            except Exception as err:
                G_LOGGER.error(f"Could not write to calibration cache: {self._cache}.\nNote: Error was: {err}")

        def free(self):
            """
            Frees all device buffers associated with this calibrator
            """
            for device_buffer in self.device_buffers.values():
                device_buffer.free()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.free()

        # IInt8LegacyCalibrator methods
        if BaseClass == trt.IInt8LegacyCalibrator:

            def get_quantile(self):
                return util.default(quantile, 0.5)

            def get_regression_cutoff(self):
                return util.default(regression_cutoff, 0.5)

            def read_histogram_cache(self, length):
                pass

            def write_histogram_cache(self, ptr, length):
                pass

        # IInt8Calibrator methods
        if BaseClass == trt.IInt8Calibrator:

            def get_algorithm(self):
                return util.default(algo, trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2)

        def __repr__(self):
            return util.make_repr(
                "Calibrator",
                data_loader,
                cache=cache,
                BaseClass=BaseClass,
                batch_size=batch_size,
                quantile=quantile,
                regression_cutoff=regression_cutoff,
                algo=algo,
            )[0]

    return CalibratorClass()
