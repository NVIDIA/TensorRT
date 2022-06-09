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

from collections import OrderedDict

from polygraphy import config, mod, util
from polygraphy.common.interface import TypedDict, TypedList
from polygraphy.json import Decoder, Encoder, add_json_methods, load_json, save_json
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")


class LazyNumpyArray:
    """
    Represents a lazily loaded NumPy array.
    For example, large NumPy arrays may be serialized to temporary files on the disk
    to save memory.
    """

    def __init__(self, arr):
        """
        Args:
            arr (np.ndarray): The NumPy array.
        """
        self.arr = None
        self.tmpfile = None
        if config.ARRAY_SWAP_THRESHOLD_MB >= 0 and arr.nbytes > (config.ARRAY_SWAP_THRESHOLD_MB << 20):
            self.tmpfile = util.NamedTemporaryFile(suffix=".json")
            G_LOGGER.extra_verbose(
                f"Evicting large array ({arr.nbytes / 1024.0 ** 2:.3f} MiB) from memory and saving to {self.tmpfile.name}"
            )
            save_json(arr, self.tmpfile.name)
        else:
            self.arr = arr

    def numpy(self):
        """
        Get the NumPy array, deserializing from the disk if it was stored earlier.

        Returns:
            np.ndarray: The NumPy array
        """
        if self.arr is not None:
            return self.arr

        assert self.tmpfile is not None, "Path and NumPy array cannot both be None!"
        return load_json(self.tmpfile.name)


@Encoder.register(LazyNumpyArray)
def encode(lazy_arr):
    return {
        "values": lazy_arr.numpy(),
    }


@Decoder.register(LazyNumpyArray)
def decode(dct):
    return LazyNumpyArray(dct["values"])


@mod.export()
class IterationResult(TypedDict(lambda: str, lambda: LazyNumpyArray)):
    """
    An ordered dictionary containing the result of a running a single iteration of a runner.

    This maps output names to NumPy arrays, and preserves the output ordering from the runner.

    NOTE: The ``POLYGRAPHY_ARRAY_SWAP_THRESHOLD_MB`` environment variable can be set to enable
    the arrays to be swapped to the disk.

    Also includes additional fields indicating the name of the runner which produced the
    outputs, and the time required to do so.
    """

    @staticmethod
    def _to_lazy(nparray):
        if isinstance(nparray, LazyNumpyArray):
            return nparray
        return LazyNumpyArray(nparray)

    @staticmethod
    def _to_lazy_dict(nparray_dict):
        if nparray_dict is None:
            return None

        # Converts a Dict[str, np.ndarray] to a Dict[str, LazyNumpyArray]
        lazy = OrderedDict()
        for name, out in nparray_dict.items():
            lazy[name] = IterationResult._to_lazy(out)
        return lazy

    def __init__(self, outputs=None, runtime=None, runner_name=None):
        """
        Args:
            outputs (Dict[str, np.array]): The outputs of this iteration, mapped to their names.

            runtime (float):
                    The time required for this iteration, in seconds.
                    Only used for logging purposes.
            runner_name (str):
                    The name of the runner that produced this output.
                    If this is omitted, a default name is generated.
        """
        if outputs and config.ARRAY_SWAP_THRESHOLD_MB < 0:
            total_size_gb = sum(arr.nbytes for arr in outputs.values() if isinstance(arr, np.ndarray)) / (1024.0 ** 3)
            if total_size_gb >= 1:
                G_LOGGER.warning(
                    f"It looks like the outputs of this network are very large ({total_size_gb:.3f} GiB).\n"
                    "To reduce memory usage, you may want to allow Polygraphy to swap these arrays to the disk "
                    "using the POLYGRAPHY_ARRAY_SWAP_THRESHOLD_MB environment variable."
                )

        super().__init__(IterationResult._to_lazy_dict(outputs))
        self.runtime = runtime
        self.runner_name = util.default(runner_name, "custom_runner")

    # Convenience methods to preserve np.ndarray in the interface.
    def update(self, other):
        return super().update(IterationResult._to_lazy_dict(other))

    def __setitem__(self, name, arr):
        return super().__setitem__(name, IterationResult._to_lazy(arr))

    def values(self):
        for arr in super().values():
            yield arr.numpy()

    def items(self):
        for name, arr in super().items():
            yield name, arr.numpy()

    def __getitem__(self, name):
        return super().__getitem__(name).numpy()

    def __eq__(self, other):
        if self.runtime != other.runtime or self.runner_name != other.runner_name:
            return False

        for key, val in self.items():
            if key not in other:
                return False

            if not np.array_equal(val, other[key]):
                return False

        return True


@Encoder.register(IterationResult)
def encode(iter_result):
    return {
        "outputs": iter_result.dct,
        "runtime": iter_result.runtime,
        "runner_name": iter_result.runner_name,
    }


@Decoder.register(IterationResult)
def decode(dct):
    return IterationResult(outputs=dct["outputs"], runtime=dct["runtime"], runner_name=dct["runner_name"])


@mod.export()
@add_json_methods("inference results")
class RunResults(TypedList(lambda: tuple)):
    """
    Maps runners to per-iteration outputs (in the form of a ``List[IterationResult]``).

    For example, if ``results`` is an instance of ``RunResults()``, then
    to access the outputs of the first iteration from a specified runner, do:
    ::

        iteration = 0
        runner_name = "trt-runner"
        outputs = results[runner_name][iteration]

        # `outputs` is a `Dict[str, np.ndarray]`


    Note: Technically, this is a ``List[Tuple[str, List[IterationResult]]]``, but includes
    helpers that make it behave like an OrderedDict that can contain duplicates.
    """

    def items(self):
        """
        Creates a generator that yields ``Tuple[str, List[IterationResult]]`` - runner names
        and corresponding outputs.
        """
        for name, iteration_results in self.lst:
            yield name, iteration_results

    def keys(self):
        """
        Creates a generator that yields runner names (str).
        """
        for name, _ in self.lst:
            yield name

    def values(self):
        """
        Creates a generator that yields runner outputs (List[IterationResult]).
        """
        for _, iteration_results in self.lst:
            yield iteration_results

    def update(self, other):
        """
        Updates the results stored in this instance.

        Args:
            other (Union[Dict[str, List[IterationResult]], RunResults]):
                    A dictionary or RunResults instance from which to update this one.
        """
        for name, iteration_results in other.items():
            self.lst[name] = iteration_results
        return self

    def add(self, out_list, runtime=None, runner_name=None):
        """
        A helper to create a ``List[IterationResult]`` and map it to the specified runner_name.

        This method cannot be used to modify an existing entry.

        Calling this method is equivalent to:
        ::

            results[runner_name] = []
            for out in out_list:
                results[runner_name].append(IterationResult(out, runtime, runner_name))

        Args:
            out_list (List[Dict[str, np.array]]):
                One or more set of outputs where each output is a dictionary
                of output names mapped to NumPy arrays.

            runtime (float):
                    The time required for this iteration, in seconds.
                    Only used for logging purposes.
            runner_name (str):
                    The name of the runner that produced this output.
                    If this is omitted, a default name is generated.
        """
        runner_name = util.default(runner_name, "custom_runner")
        iter_results = [IterationResult(out, runtime, runner_name) for out in out_list]
        self[runner_name] = iter_results

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.lst[key]

        for name, iteration_results in self.lst:
            if name == key:
                return iteration_results

        G_LOGGER.critical(
            f"{key:35} does not exist in this RunResults instance. Note: Available runners: {list(self.keys())}"
        )

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.lst[key] = value
            return

        for index, name in enumerate(self.keys()):
            if name == key:
                self.lst[index] = (key, value)
                break
        else:
            self.append((key, value))

    def __contains__(self, val):
        if isinstance(val, str) or isinstance(val, bytes):
            return val in list(self.keys())
        return val in self.lst

    def __eq__(self, other):
        for (r0, its0), (r1, its1) in zip(self.lst, other.lst):
            if r0 != r1:
                return False

            if its0 != its1:
                return False
        return True


@Encoder.register(RunResults)
def encode(results):
    return {"lst": results.lst}


@Decoder.register(RunResults)
def decode(dct):
    return RunResults(list(map(tuple, dct["lst"])))


@mod.export()
class AccuracyResult(TypedDict(lambda: tuple, lambda: list)):
    """
    An ordered dictionary including details about the result of ``Comparator.compare_accuracy``.

    More specifically, it is an ``OrderedDict[Tuple[str, str], List[OrderedDict[str, bool]]]`` which maps a runner
    pair (a tuple containing both runner names) to a list of dictionaries of booleans (or anything that can be
    converted into a boolean, such as an ``OutputCompareResult``), indicating whether there was a match in the outputs of
    the corresponding iteration. The ``List[OrderedDict[str, bool]]`` is constructed from the dictionaries returned
    by ``compare_func`` in ``compare_accuracy``.

    For example, to see if there's a match between ``runner0`` and
    ``runner1`` during the 1st iteration for an output called ``output0``:
    ::

        runner_pair = ("runner0", "runner1")
        iteration = 0
        output_name = "output0"
        match = bool(accuracy_result[runner_pair][iteration][output_name])

    If there's a mismatch, you can inspect the outputs from
    the results of ``Comparator.run()``, assumed here to be called ``run_results``:
    ::

        runner0_output = run_results["runner0"][iteration][output_name]
        runner1_output = run_results["runner1"][iteration][output_name]
    """

    def __bool__(self):
        """
        Whether all outputs matched for every iteration.
        You can use this function to avoid manually checking each output. For example:
        ::

            if accuracy_result:
                print("All matched!")

        Returns:
            bool
        """
        return all([bool(match) for outs in self.values() for out in outs for match in out.values()])

    def _get_runner_pair(self, runner_pair):
        return util.default(runner_pair, list(self.keys())[0])

    def percentage(self, runner_pair=None):
        """
        Returns the percentage of iterations that matched for the given pair of runners,
        expressed as a decimal between 0.0 and 1.0.

        Always returns 1.0 when the number of iterations is 0, or when there are no runner comparisons.

        Args:
            runner_pair (Tuple[str, str]):
                    A pair of runner names describing which runners to check.
                    Defaults to the first pair in the dictionary.
        """
        if not list(self.keys()):
            return 1.0  # No data in this result.

        matched, _, total = self.stats(runner_pair)
        if not total:
            return 1.0  # No iterations
        return float(matched) / float(total)

    def stats(self, runner_pair=None):
        """
        Returns the number of iterations that matched, mismatched, and the total number of iterations.

        Args:
            runner_pair (Tuple[str, str]):
                    A pair of runner names describing which runners to check.
                    Defaults to the first pair in the dictionary.

        Returns:
            Tuple[int, int, int]: Number of iterations that matched, mismatched, and total respectively.
        """
        runner_pair = self._get_runner_pair(runner_pair)
        outs = self[runner_pair]
        matched = sum([all([match for match in out.values()]) for out in outs])
        total = len(outs)
        return matched, total - matched, total
