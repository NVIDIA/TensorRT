/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file contains all int8 calibration related docstrings, since these are typically too long to keep in the binding
// code.
#pragma once

namespace tensorrt
{
namespace CalibrationAlgoTypeDoc
{
constexpr const char* descr = R"trtdoc(
    Version of calibration algorithm to use.
)trtdoc";
} // namespace CalibrationAlgoTypeDoc

namespace IInt8CalibratorDoc
{
constexpr const char* descr = R"trtdoc(
    Application-implemented interface for calibration. Calibration is a step performed by the builder when deciding suitable scale factors for 8-bit inference. It must also provide a method for retrieving representative images which the calibration process can use to examine the distribution of activations. It may optionally implement a method for caching the calibration result for reuse on subsequent runs.

    To implement a custom calibrator, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyCalibrator(trt.IInt8Calibrator):
            def __init__(self):
                trt.IInt8Calibrator.__init__(self)

    :ivar batch_size: :class:`int` The batch size used for calibration batches.
    :ivar algorithm: :class:`CalibrationAlgoType` The algorithm used by this calibrator.
)trtdoc";

constexpr const char* get_batch_size = R"trtdoc(
    Get the batch size used for calibration batches.

    :returns: The batch size.
)trtdoc";

constexpr const char* get_algorithm = R"trtdoc(
    Get the algorithm used by this calibrator.

    :returns: The algorithm used by this calibrator.
)trtdoc";

constexpr const char* get_batch = R"trtdoc(
    Get a batch of input for calibration. The batch size of the input must match the batch size returned by :func:`get_batch_size` .

    A possible implementation may look like this:
    ::

        def get_batch(names):
            try:
                # Assume self.batches is a generator that provides batch data.
                data = next(self.batches)
                # Assume that self.device_input is a device buffer allocated by the constructor.
                cuda.memcpy_htod(self.device_input, data)
                return [int(self.device_input)]
            except StopIteration:
                # When we're out of batches, we return either [] or None.
                # This signals to TensorRT that there is no calibration data remaining.
                return None

    :arg names: The names of the network inputs for each object in the bindings array.

    :returns: A :class:`list` of device memory pointers set to the memory containing each network input data, or an empty :class:`list` if there are no more batches for calibration. You can allocate these device buffers with pycuda, for example, and then cast them to :class:`int` to retrieve the pointer.
)trtdoc";

constexpr const char* read_calibration_cache = R"trtdoc(
    Load a calibration cache.

    Calibration is potentially expensive, so it can be useful to generate the calibration data once, then use it on subsequent builds
    of the network. The cache includes the regression cutoff and quantile values used to generate it, and will not be used if
    these do not match the settings of the current calibrator. However, the network should also be recalibrated if its structure
    changes, or the input data set changes, and it is the responsibility of the application to ensure this.

    Reading a cache is just like reading any other file in Python. For example, one possible implementation is:
    ::

        def read_calibration_cache(self):
            # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()

    :returns: A cache object or None if there is no data.
)trtdoc";

constexpr const char* write_calibration_cache = R"trtdoc(
    Save a calibration cache.

    Writing a cache is just like writing any other buffer in Python. For example, one possible implementation is:
    ::

        def write_calibration_cache(self, cache):
            with open(self.cache_file, "wb") as f:
                f.write(cache)

    :arg cache: The calibration cache to write.
)trtdoc";

} // namespace IInt8CalibratorDoc

namespace IInt8LegacyCalibratorDoc
{
constexpr const char* descr = R"trtdoc(
    Extends the :class:`IInt8Calibrator` class.
    This calibrator requires user parameterization, and is provided as a fallback option if the other calibrators yield poor results.

    To implement a custom calibrator, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyCalibrator(trt.IInt8LegacyCalibrator):
            def __init__(self):
                trt.IInt8LegacyCalibrator.__init__(self)

    :ivar quantile: :class:`float` The quantile (between 0 and 1) that will be used to select the region maximum when the quantile method is in use. See the user guide for more details on how the quantile is used.
    :ivar regression_cutoff: :class:`float` The fraction (between 0 and 1) of the maximum used to define the regression cutoff when using regression to determine the region maximum. See the user guide for more details on how the regression cutoff is used
)trtdoc";

constexpr const char* readHistogramCache = R"trtdoc(
    Load a histogram.
    Histogram generation is potentially expensive, so it can be useful to generate the histograms once, then use them when exploring
    the space of calibrations. The histograms should be regenerated if the network structure
    changes, or the input data set changes, and it is the responsibility of the application to ensure this.
    See the user guide for more details on how the regression cutoff is used

    :arg length: The length of the cached data, that should be set by the called function. If there is no data, this should be zero.

    :returns: The cache or None if there is no cache.
)trtdoc";

constexpr const char* writeHistogramCache = R"trtdoc(
    Save a histogram cache.

    :arg data: The data to cache.
    :arg length: The length in bytes of the data to cache.
)trtdoc";

constexpr const char* get_algorithm = R"trtdoc(
    Signals that this is the legacy calibrator.

    :returns: :class:`CalibrationAlgoType.LEGACY_CALIBRATION`
)trtdoc";
} // namespace IInt8LegacyCalibratorDoc

namespace IInt8EntropyCalibratorDoc
{
constexpr const char* descr = R"trtdoc(
    Extends the :class:`IInt8Calibrator` class.

    To implement a custom calibrator, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyCalibrator(trt.IInt8EntropyCalibrator):
            def __init__(self):
                trt.IInt8EntropyCalibrator.__init__(self)


    This is the Legacy Entropy calibrator. It is less complicated than the legacy calibrator and produces better results.
)trtdoc";

constexpr const char* get_algorithm = R"trtdoc(
    Signals that this is the entropy calibrator.

    :returns: :class:`CalibrationAlgoType.ENTROPY_CALIBRATION`
)trtdoc";
} // namespace IInt8EntropyCalibratorDoc

namespace IInt8EntropyCalibrator2Doc
{
constexpr const char* descr = R"trtdoc(
    Extends the :class:`IInt8Calibrator` class.

    To implement a custom calibrator, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyCalibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                trt.IInt8EntropyCalibrator2.__init__(self)

    This is the preferred calibrator. This is the required calibrator for DLA, as it supports per activation tensor scaling.
)trtdoc";

constexpr const char* get_algorithm = R"trtdoc(
    Signals that this is the entropy calibrator 2.

    :returns: :class:`CalibrationAlgoType.ENTROPY_CALIBRATION_2`
)trtdoc";
} // namespace IInt8EntropyCalibrator2Doc

namespace IInt8MinMaxCalibratorDoc
{
constexpr const char* descr = R"trtdoc(
    Extends the :class:`IInt8Calibrator` class.

    To implement a custom calibrator, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyCalibrator(trt.IInt8MinMaxCalibrator):
            def __init__(self):
                trt.IInt8MinMaxCalibrator.__init__(self)

    This is the preferred calibrator for NLP tasks for all backends. It supports per activation tensor scaling.
)trtdoc";

constexpr const char* get_algorithm = R"trtdoc(
    Signals that this is the minmax calibrator.

    :returns: :class:`CalibrationAlgoType.MINMAX_CALIBRATION`
)trtdoc";
} // namespace IInt8MinMaxCalibratorDoc

} // namespace tensorrt
