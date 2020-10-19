#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import os
from collections import OrderedDict

import tensorrt as trt
from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.util import misc
from polygraphy.util.cuda import DeviceBuffer


def Calibrator(data_loader, cache=None, BaseClass=trt.IInt8MinMaxCalibrator,
               batch_size=None):
    """
    Supplies calibration data to TensorRT to calibrate the network for INT8 inference.

    Args:
        data_loader (Generator -> OrderedDict[str, np.ndarray]):
            A generator or iterable that yields a dictionary that maps input names to input NumPy buffers.

            In case you don't know details about the inputs ahead of time, you can access the
            `input_metadata` property in your data loader, which will be set to an `TensorMetadata` instance.
            Note that this does not work for generators or lists.

            The number of calibration batches is controlled by the number of items supplied
            by the data loader.


        cache (Union[str, file-like]):
                Path or file-like object to save/load the calibration cache.
                By default, the calibration cache is not saved.
        BaseClass (type):
                The type of calibrator to inherit from.
                Defaults to trt.IInt8MinMaxCalibrator.
        batch_size (int):
                [DEPRECATED] The size of each batch provided by the data loader.
    """
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
            self.reset()
            G_LOGGER.verbose("Created calibrator [cache={:}]".format(self._cache))

            self.batch_size = misc.default_value(batch_size, 1)


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
            try:
                host_buffers = next(self.data_loader_iter)
            except StopIteration:
                if not self.num_batches:
                    G_LOGGER.warning("Calibrator data loader provided no data. Possibilities include: (1) data loader "
                                     "has no data to provide, (2) data loader was a generator, and the calibrator is being "
                                     "reused across multiple loaders (generators cannot be rewound)")
                return None
            else:
                self.num_batches += 1

            for name, host_buffer in host_buffers.items():
                if name not in self.device_buffers:
                    self.device_buffers[name] = DeviceBuffer(shape=host_buffer.shape, dtype=host_buffer.dtype)
                    G_LOGGER.verbose("Allocated: {:}".format(self.device_buffers[name]))
                    if self.num_batches > 1:
                        G_LOGGER.warning("The calibrator data loader provided an extra input ({:}) compared to the last set of inputs.\n"
                                         "Should this input be removed, or did you accidentally omit an input before?".format(name))

                device_buffer = self.device_buffers[name]
                device_buffer.copy_from(host_buffer)
            return [device_buffer.address() for device_buffer in self.device_buffers.values()]


        def read_calibration_cache(self):
            def load_from_cache():
                if self._cache is None:
                    return None

                try:
                    if self._cache.seekable():
                        self._cache.seek(0)
                    return self._cache.read()
                except AttributeError:
                    if os.path.exists(self._cache):
                        G_LOGGER.info("Reading calibration cache from: {:}".format(self._cache), mode=LogMode.ONCE)
                        with open(self._cache, "rb") as f:
                            return f.read()
                except:
                    # Cache is not readable
                    return None


            if not self.has_cached_scales:
                self.cache_contents = load_from_cache()
                if not self.cache_contents:
                    G_LOGGER.warning("Calibration cache was provided, but is empty. Will regenerate scales by running calibration.", mode=LogMode.ONCE)
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
                if self._cache.seekable():
                    self._cache.seek(0)
                bytes_written = self._cache.write(self.cache_contents)
                if bytes_written != len(self.cache_contents):
                    G_LOGGER.warning("Could not write entire cache. Note: cache contains {:} bytes, but only "
                                        "{:} bytes were written".format(len(self.cache_contents), bytes_written))
            except AttributeError:
                G_LOGGER.info("Writing calibration cache to: {:}".format(self._cache))
                with open(self._cache, "wb") as f:
                    f.write(self.cache_contents)
            except:
                # Cache is not writable
                return
            else:
                self._cache.flush()


        def free(self):
            """
            Free the device buffers allocated for this calibrator.
            """
            for device_buffer in self.device_buffers.values():
                device_buffer.free()


    return CalibratorClass()
