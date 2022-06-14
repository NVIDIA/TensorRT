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
import contextlib
from collections import OrderedDict

from polygraphy import cuda, mod, util
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
        data_loader (Generator -> OrderedDict[str, Union[numpy.ndarray, DeviceView, int]]):
            A generator or iterable that yields a dictionary that maps input names to NumPy
            arrays, Polygraphy DeviceViews, or GPU pointers.

            In case you don't know details about the inputs ahead of time, you can access the
            `input_metadata` property in your data loader, which will be set to an ``TensorMetadata`` instance.
            Note that this does not work for generators or lists.

            The number of calibration batches is controlled by the number of items supplied
            by the data loader.


        cache (Union[str, file-like]):
                Path or file-like object to save/load the calibration cache.
                By default, the calibration cache is not saved.
        BaseClass (type):
                The type of calibrator to inherit from.
                Defaults to ``trt.IInt8MinMaxCalibrator``.
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
    BaseClass = util.default(BaseClass, trt.IInt8MinMaxCalibrator)

    class CalibratorClass(BaseClass):
        """
        Calibrator that supplies calibration data to TensorRT to calibrate the network for INT8 inference.
        """

        def __init__(self):
            # Must explicitly initialize parent for any trampoline class! Will mysteriously segfault without this.
            BaseClass.__init__(self)

            self.is_active = False

            self.data_loader = data_loader
            self._cache = cache
            self.device_buffers = OrderedDict()
            self.reset()
            G_LOGGER.verbose("Created calibrator [cache={:}]".format(self._cache))

            self.batch_size = util.default(batch_size, 1)

            # The function that constructed this instance
            self.make_func = Calibrator

        def reset(self, input_metadata=None):
            """
            Reset this calibrator for reuse.
            The calibrator will clear any dynamic ranges cached from previous calibration runs, and will
            attempt to rewind the data loader (note that generators cannot be rewound).

            Args:
                input_metadata (TensorMetadata):
                        Mapping of input names to their data types and shapes.
                        Passed along to the data loader if provided. Generally should not be required
                        unless using Polygraphy's included `DataLoader` for this calibrator.
            """
            if input_metadata is not None:
                with contextlib.suppress(AttributeError):
                    self.data_loader.input_metadata = input_metadata

            # Attempt to reset data loader
            self.data_loader_iter = iter(self.data_loader)
            self.num_batches = 0

            # Make sure calibrator will check the cache again when reset.
            self.cache_contents = None
            self.has_cached_scales = False

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if not self.is_active:
                G_LOGGER.error(
                    "Calibrator must be activated prior to use. Please use a context manager. "
                    "For example:\nwith calibrator:\n\t# Use calibrator here"
                )
                return None

            try:
                buffers = next(self.data_loader_iter)
            except StopIteration:
                if not self.num_batches:
                    G_LOGGER.error(
                        "Calibrator data loader provided no data.\nPossible reasons for this include:\n(1) data loader "
                        "has no data to provide\n(2) data loader was a generator, and the calibrator is being "
                        "used multiple times (generators cannot be rewound)"
                    )
                return None
            else:
                self.num_batches += 1

            if not util.check_dict_contains(buffers, names, dict_name="calibration data", log_func=G_LOGGER.error):
                return None

            ptrs = []
            for name in names:
                buf = buffers[name]

                if isinstance(buf, cuda.DeviceView):
                    ptrs.append(buf.ptr)
                elif isinstance(buf, np.ndarray):
                    if name not in self.device_buffers:
                        self.device_buffers[name] = cuda.DeviceArray(shape=buf.shape, dtype=buf.dtype)
                        G_LOGGER.verbose("Allocated: {:}".format(self.device_buffers[name]))

                    ptrs.append(self.device_buffers[name].copy_from(buf).ptr)
                elif isinstance(buf, int):
                    ptrs.append(buf)
                else:
                    G_LOGGER.error(
                        "Calibration data loader provided an unrecognized type: {:} for input: {:}.\n"
                        "Please provide either a NumPy array, Polygraphy DeviceView, or GPU pointer. ".format(
                            type(buf).__name__, name
                        )
                    )
                    return None

            return ptrs

        def read_calibration_cache(self):
            def load_from_cache():
                if self._cache is None or not util.get_file_size(self._cache):
                    return None

                try:
                    return util.load_file(self._cache, description="calibration cache")
                except Exception as err:
                    G_LOGGER.error(
                        "Could not read from calibration cache: {:}\nNote: Error was: {:}".format(self._cache, err)
                    )
                    return None

            # Only attempt to read from the cache once.
            if self.has_cached_scales:
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
            else:
                self.has_cached_scales = True

            return self.cache_contents

        def write_calibration_cache(self, cache):
            self.cache_contents = cache.tobytes()
            self.has_cached_scales = True

            if self._cache is None:
                return

            try:
                util.save_file(contents=self.cache_contents, dest=self._cache, description="calibration cache")
            except Exception as err:
                G_LOGGER.error(
                    "Could not write to calibration cache: {:}.\nNote: Error was: {:}".format(self._cache, err)
                )

        def __enter__(self):
            self.is_active = True
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.is_active = False
            for device_buffer in self.device_buffers.values():
                device_buffer.free()

        # IInt8LegacyCalibrator methods
        def get_quantile(self):
            return util.default(quantile, 0.5)

        def get_regression_cutoff(self):
            return util.default(regression_cutoff, 0.5)

        def read_histogram_cache(self, length):
            pass

        def write_histogram_cache(self, ptr, length):
            pass

        # IInt8Calibrator methods
        def get_algorithm(self):
            return util.default(algo, trt.CalibrationAlgoType.MINMAX_CALIBRATION)

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
