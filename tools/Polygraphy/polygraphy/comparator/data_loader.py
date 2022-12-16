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

from polygraphy import constants, func, mod, util
from polygraphy.exception import PolygraphyException
from polygraphy.json import save_json
from polygraphy.logger import G_LOGGER, LogMode

np = mod.lazy_import("numpy")


@mod.export()
class DataLoader:
    """
    Generates synthetic input data.
    """

    def __init__(
        self, seed=None, iterations=None, input_metadata=None, int_range=None, float_range=None, val_range=None
    ):
        """
        Args:
            seed (int):
                    The seed to use when generating random inputs.
                    Defaults to ``util.constants.DEFAULT_SEED``.
            iterations (int):
                    The number of iterations for which to supply data.
                    Defaults to 1.
            input_metadata (TensorMetadata):
                    A mapping of input names to their corresponding shapes and data types.
                    This will be used to determine what shapes to supply for inputs with dynamic shape, as
                    well as to set the data type of the generated inputs.
                    If either dtype or shape are None, then the value will be automatically determined.
                    For input shape tensors, i.e. inputs whose *value* describes a shape in the model, the
                    provided shape will be used to populate the values of the inputs, rather than to determine
                    their shape.
            val_range (Union[Tuple[number], Dict[str, Tuple[number]]]):
                    A tuple containing exactly 2 numbers, indicating the minimum and maximum values (inclusive)
                    the data loader should generate.
                    If either value in the tuple is None, the default will be used for that value.
                    If None is provided instead of a tuple, then the default values will be used for both the
                    minimum and maximum.
                    This can be specified on a per-input basis using a dictionary. In that case,
                    use an empty string ("") as the key to specify default range for inputs not explicitly listed.
                    Defaults to (0.0, 1.0).

            int_range (Tuple[int]):
                    [DEPRECATED - Use val_range instead]
                    A tuple containing exactly 2 integers, indicating the minimum and maximum integer values (inclusive)
                    the data loader should generate. If either value in the tuple is None, the default will be used
                    for that value.
                    If None is provided instead of a tuple, then the default values will be used for both the
                    minimum and maximum.
            float_range (Tuple[float]):
                    [DEPRECATED - Use val_range instead]
                    A tuple containing exactly 2 floats, indicating the minimum and maximum float values (inclusive)
                    the data loader should generate. If either value in the tuple is None, the default will be used
                    for that value.
                    If None is provided instead of a tuple, then the default values will be used for both the
                    minimum and maximum.
        """

        def default_tuple(tup, default):
            if tup is None or (not isinstance(tup, tuple) and not isinstance(tup, list)):
                return default
            new_tup = []
            for elem, default_elem in zip(tup, default):
                new_tup.append(util.default(elem, default_elem))
            return tuple(new_tup)

        self.seed = util.default(seed, constants.DEFAULT_SEED)
        self.iterations = util.default(iterations, 1)
        self.user_input_metadata = util.default(input_metadata, {})

        self._int_range_set = int_range is not None
        if self._int_range_set:
            mod.warn_deprecated("The int_range parameter in DataLoader", "val_range", remove_in="0.50.0")
        self._int_range = default_tuple(int_range, (1, 25))

        self._float_range_set = float_range is not None
        if self._float_range_set:
            mod.warn_deprecated("The float_range parameter in DataLoader", "val_range", remove_in="0.50.0")
        self._float_range = default_tuple(float_range, (-1.0, 1.0))

        self.input_metadata = None
        self.default_val_range = default_tuple(val_range, (0.0, 1.0))
        self.val_range = util.default(val_range, self.default_val_range)

        if self.user_input_metadata:
            G_LOGGER.info(
                f"Will generate inference input data according to provided TensorMetadata: {self.user_input_metadata}"
            )

    def __repr__(self):
        return util.make_repr(
            "DataLoader",
            seed=self.seed,
            iterations=self.iterations,
            input_metadata=self.user_input_metadata or None,
            int_range=self._int_range,
            float_range=self._float_range,
            val_range=self.val_range,
        )[0]

    def _get_range(self, name, cast_type):
        if cast_type == int and self._int_range_set:
            return self._int_range
        elif cast_type == float and self._float_range_set:
            return self._float_range

        tup = util.value_or_from_dict(self.val_range, name, self.default_val_range)
        return tuple(cast_type(val) for val in tup)

    def __getitem__(self, index):
        """
        Generates random input data.

        May update the DataLoader's `input_metadata` attribute.

        Args:
            index (int):
                    Since this class behaves like an iterable, it takes an index parameter.
                    Generated data is guaranteed to be the same for the same index.

        Returns:
            OrderedDict[str, numpy.ndarray]: A mapping of input names to input numpy buffers.
        """
        if index >= self.iterations:
            raise IndexError()

        G_LOGGER.verbose(f"Generating data using numpy seed: {self.seed + index}")
        rng = np.random.RandomState(self.seed + index)

        def get_static_shape(name, shape):
            static_shape = shape
            if util.is_shape_dynamic(shape):
                if shape.min is not None:
                    static_shape = shape.min
                elif shape.max is not None:
                    static_shape = shape.max
                else:
                    static_shape = util.override_dynamic_shape(shape)

                if static_shape != shape:
                    if not util.is_valid_shape_override(static_shape, shape):
                        G_LOGGER.critical(
                            f"Input tensor: {name} | Cannot override original shape: {shape} to {static_shape}"
                        )
                    G_LOGGER.warning(
                        f"Input tensor: {name} [shape={shape}] | Will generate data of shape: {static_shape}.\n"
                        f"If this is incorrect, please provide a custom data loader.",
                        mode=LogMode.ONCE,
                    )
            return static_shape

        # Whether the user provided the values for a shape tensor input,
        # rather than the shape of the input.
        # If the shape is 1D, and has a value equal to the rank of the provided default shape, it is
        # likely to be a shape tensor, and so its value, not shape, should be overriden.
        def is_shape_tensor(name, dtype):
            if name not in self.input_metadata or name not in self.user_input_metadata:
                return False

            _, shape = self.input_metadata[name]
            is_shape = np.issubdtype(dtype, np.integer) and (not util.is_shape_dynamic(shape)) and (len(shape) == 1)

            user_shape = self.user_input_metadata[name].shape
            is_shape &= len(user_shape) == shape[0]
            is_shape &= not util.is_shape_dynamic(user_shape)  # Shape of shape cannot be dynamic.
            return is_shape

        def generate_buffer(name, dtype, shape):
            if is_shape_tensor(name, dtype):
                buffer = np.array(shape, dtype=dtype)
                G_LOGGER.info(
                    f"Assuming {name} is a shape tensor. Setting input values to: {buffer}. "
                    "If these values are not correct, please set it correctly in 'input_metadata' or by providing --input-shapes",
                    mode=LogMode.ONCE,
                )
            elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
                imin, imax = self._get_range(name, cast_type=int if np.issubdtype(dtype, np.integer) else bool)
                G_LOGGER.verbose(
                    f"Input tensor: {name} | Generating input data in range: [{imin}, {imax}]",
                    mode=LogMode.ONCE,
                )
                # high is 1 greater than the max int drawn.
                buffer = rng.randint(low=imin, high=imax + 1, size=shape, dtype=dtype)
            else:
                fmin, fmax = self._get_range(name, cast_type=float)
                G_LOGGER.verbose(
                    f"Input tensor: {name} | Generating input data in range: [{fmin}, {fmax}]",
                    mode=LogMode.ONCE,
                )

                # Special handling for infinite lower/upper bounds
                # Without this, two inifinities will collapse into a NaN, resulting in no inifinities
                # in the final output.
                scale = fmax - fmin
                shift = fmin
                if util.is_inf(fmin):
                    scale = fmin
                    shift = 0
                if util.is_inf(fmax):
                    scale = fmax

                buffer = (rng.random_sample(size=shape) * scale + shift).astype(dtype)

            buffer = np.array(buffer)  # To handle scalars, since the above functions return a float if shape is ().
            return buffer

        if self.input_metadata is None and self.user_input_metadata is not None:
            self.input_metadata = self.user_input_metadata

        buffers = OrderedDict()
        for name, (dtype, shape) in self.input_metadata.items():
            if name in self.user_input_metadata:
                user_dtype, user_shape = self.user_input_metadata[name]

                dtype = util.default(user_dtype, dtype)
                is_valid_shape_override = user_shape is not None and util.is_valid_shape_override(user_shape, shape)

                if util.is_shape_dynamic(user_shape):
                    G_LOGGER.warning(
                        f"Input tensor: {name} [shape={shape}] | Provided input shape: {user_shape} is dynamic.\nDynamic shapes cannot be used to generate inference data. Will use default shape instead.\nTo avoid this, please provide a fixed shape to the data loader. "
                    )
                elif not is_valid_shape_override and not is_shape_tensor(name, dtype):
                    G_LOGGER.warning(
                        f"Input tensor: {name} [shape={shape}] | Cannot use provided custom shape: {user_shape} to override tensor shape. Will use default shape instead.",
                        mode=LogMode.ONCE,
                    )
                else:
                    shape = util.default(user_shape, shape)

            static_shape = get_static_shape(name, shape)
            buffers[name] = generate_buffer(name, dtype, shape=static_shape)

        # Warn about unused metadata
        for name in self.user_input_metadata.keys():
            if name not in self.input_metadata:
                msg = f"Input tensor: {name} | Metadata was provided, but the input does not exist in one or more runners."
                close_match = util.find_str_in_iterable(name, self.input_metadata.keys())
                if close_match:
                    msg += f"\nMaybe you meant to set: {close_match}?"
                G_LOGGER.warning(msg)

        # Warn about unused val_range
        if not isinstance(self.val_range, tuple):
            util.check_sequence_contains(
                self.val_range.keys(),
                list(self.input_metadata.keys()) + [""],
                name="val_range",
                log_func=G_LOGGER.warning,
                check_missing=False,
            )

        return buffers


# Caches data loaded by a DataLoader for use across multiple runners.
class DataLoaderCache:
    def __init__(self, data_loader, save_inputs_path=None):
        self.data_loader = data_loader
        self.cache = []  # List[OrderedDict[str, numpy.ndarray]]
        self.save_inputs_path = save_inputs_path

    @func.constantmethod
    def __getitem__(self, iteration):
        """
        Load the specified iteration from the cache if present, or load it from the data loader.

        Args:
            iteration (int): The iteration whose data to retrieve.
        """
        if iteration >= len(self.cache):
            raise IndexError()

        # Attempts to match existing input buffers to the requested input_metadata
        def coerce_cached_input(index, name, dtype, shape):
            cached_feed_dict = self.cache[iteration]
            cached_name = util.find_str_in_iterable(name, cached_feed_dict.keys(), index)
            if cached_name is None:
                G_LOGGER.critical(f"Input tensor: {name} | Does not exist in the data loader cache.")

            if cached_name != name:
                G_LOGGER.warning(
                    f"Input tensor: {name} | Buffer name ({cached_name}) does not match expected input name ({name})."
                )

            buffer = cached_feed_dict[cached_name]

            if dtype != buffer.dtype:
                G_LOGGER.warning(
                    f"Input tensor: {name} | Buffer dtype ({buffer.dtype}) does not match expected input dtype ({np.dtype(dtype).name}), attempting to cast. "
                )

                type_info = None
                if np.issubdtype(dtype, np.integer):
                    type_info = np.iinfo(np.dtype(dtype))
                elif np.issubdtype(dtype, np.floating):
                    type_info = np.finfo(np.dtype(dtype))

                if type_info is not None and np.any((buffer < type_info.min) | (buffer > type_info.max)):
                    G_LOGGER.warning(
                        f"Some values in this input are out of range of {dtype}. Unexpected behavior may ensue!"
                    )
                buffer = buffer.astype(dtype)

            if not util.is_valid_shape_override(buffer.shape, shape):
                G_LOGGER.warning(
                    f"Input tensor: {name} | Buffer shape ({buffer.shape}) does not match expected input shape ({shape}), attempting to transpose/reshape. "
                )
                buffer = util.try_match_shape(buffer, shape)

            if buffer.dtype != dtype or not util.is_valid_shape_override(buffer.shape, shape):
                G_LOGGER.critical(
                    f"Input tensor: {name} | Cannot reuse input data due to mismatch in shape or data type.\nNote: Cached input: [dtype={buffer.dtype}, shape={buffer.shape}], Requested input: [dtype={dtype}, shape={shape}]"
                )
            return buffer

        feed_dict = OrderedDict()

        # Reload from data loader if needed
        data_loader_feed_dict = None

        for index, (name, (dtype, shape)) in enumerate(self.input_metadata.items()):
            try:
                buffer = coerce_cached_input(index, name, dtype, shape)
            except PolygraphyException:
                G_LOGGER.warning(
                    f"Could not use buffer previously cached from data loader for input: {name}. Attempting to reload inputs from the data loader.\nNote that this will only work if the data loader supports random access.\nPlease refer to warnings above for details on why the previously generated input buffer didn't work. "
                )
                try:
                    if data_loader_feed_dict is None:
                        data_loader_feed_dict = self.data_loader[iteration]
                    buffer = data_loader_feed_dict[name]
                except:
                    G_LOGGER.critical(
                        "Could not reload inputs from data loader. Are the runners running the same model? "
                        "If not, please rewrite the data loader to support random access."
                    )
            feed_dict[name] = buffer

        return feed_dict

    def set_input_metadata(self, input_metadata):
        """
        Set the input metadata for the data loader.

        Args:
            input_metadata (TensorMetadata):
                    Input Metadata, including shape and type information. The cache may attempt to transform inputs to
                    match the specified input_metadata when data already in the cache does not exactly match.
        """
        self.input_metadata = input_metadata
        with contextlib.suppress(AttributeError):
            self.data_loader.input_metadata = input_metadata

        if not self.cache:
            G_LOGGER.verbose("Loading inputs from data loader")
            self.cache = list(self.data_loader)

            def _is_feed_dict(inp):
                try:
                    for name, arr in inp.items():
                        if not isinstance(name, str) or not isinstance(arr, np.ndarray):
                            return False
                except:
                    return False
                else:
                    return True

            if not self.cache:
                G_LOGGER.warning("Data loader did not yield any input data.")
            elif not _is_feed_dict(self.cache[0]):
                G_LOGGER.critical(
                    f"Data loader returned an object that cannot be recognized as a feed_dict (Dict[str, np.ndarray]):"
                    f"\nNote: The object was:\n{self.cache[0]}.\n"
                    f"\nHint: If this is a `RunReults` object (e.g. generated with `--save-outputs`), try using the "
                    f"`data to-input` tool to convert it to a feed_dict compatible format. "
                )

            # Only save inputs the first time the cache is generated
            if self.save_inputs_path is not None:
                save_json(self.cache, self.save_inputs_path, "inference input data")
