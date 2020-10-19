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
from collections import OrderedDict

import numpy as np
from polygraphy.common.constants import DEFAULT_SEED
from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.util import misc


class BaseDataLoader(object):
    """
    Responsible for fetching or generating input data for runners.
    """
    def __getitem__(self, index):
        """
        Fetches or generates inputs.

        Args:
            index (int): The index of inputs to fetch. For any given index, the inputs should always be the same.

        Vars:
            input_metadata (TensorMetadata):
                    Mapping of input names to their data types and shapes.
                    This is set by the Comparator when the data loader is used.

        Returns:
            OrderedDict[str, np.ndarray]: Mapping of input names to NumPy buffers containing data.
        """
        raise NotImplementedError("BaseDataLoader is an abstract class")


class DataLoader(BaseDataLoader):
    def __init__(self, seed=None, iterations=None, input_metadata=None,
                 int_range=None, float_range=None):
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
            int_range (Tuple[int]):
                    A tuple containing exactly 2 integers, indicating the minimum and maximum integer values (inclusive)
                    the data loader should generate. If either value in the tuple is None, the default will be used
                    for that value.
                    If None is provided instead of a tuple, then the default values will be used for both the
                    minimum and maximum.
            float_range (Tuple[float]):
                    A tuple containing exactly 2 floats, indicating the minimum and maximum float values (inclusive)
                    the data loader should generate. If either value in the tuple is None, the default will be used
                    for that value.
                    If None is provided instead of a tuple, then the default values will be used for both the
                    minimum and maximum.
        """
        def default_tuple(tup, default):
            if tup is None:
                return default
            new_tup = []
            for elem, default_elem in zip(tup, default):
                new_tup.append(misc.default_value(elem, default_elem))
            return tuple(new_tup)

        self.seed = misc.default_value(seed, DEFAULT_SEED)
        self.iterations = misc.default_value(iterations, 1)
        self.user_input_metadata = misc.default_value(input_metadata, {})
        self.int_range = default_tuple(int_range, (1, 25))
        self.float_range = default_tuple(float_range, (-1.0, 1.0))
        self.input_metadata = None


    def __getitem__(self, index):
        """
        Randomly generates input data.

        Args:
            index (int):
                    Since this class behaves like an iterable, it takes an index parameter.
                    Generated data is guaranteed to be the same for the same index.

            Returns:
                OrderedDict[str, np.ndarray]: A mapping of input names to input numpy buffers.
        """
        if index >= self.iterations:
            raise IndexError()

        G_LOGGER.verbose("Generating data using numpy seed: {:}".format(self.seed + index))
        rng = np.random.RandomState(self.seed + index)


        def get_static_shape(name, shape):
            static_shape = shape
            if misc.is_shape_dynamic(shape):
                static_shape = misc.override_dynamic_shape(shape)
                if static_shape != shape and name not in self.user_input_metadata:
                    if not misc.is_valid_shape_override(static_shape, shape):
                        G_LOGGER.critical("Input tensor: {:24} | Cannot override original shape: {:} to {:}".format(name, shape, static_shape))
                    G_LOGGER.warning("Input tensor: {:24} | Adjusted shape: {:} to: {:}. If this is incorrect, please set input_metadata "
                                     "or provide a custom data loader.".format(name, shape, static_shape), mode=LogMode.ONCE)
            return static_shape


        # Whether the user provided the values for a shape tensor input,
        # rather than the shape of the input.
        # If the shape is 1D, and has a value equal to the rank of the provided default shape, it is
        # likely to be a shape tensor, and so its value, not shape, should be overriden.
        def is_shape_tensor(name, dtype):
            if name not in self.input_metadata or name not in self.user_input_metadata:
                return False

            _, shape = self.input_metadata[name]
            is_shape = np.issubdtype(dtype, np.integer) and (not misc.is_shape_dynamic(shape)) and (len(shape) == 1)

            user_shape = self.user_input_metadata[name][1]
            is_shape &= len(user_shape) == shape[0]
            # Can't have negative values in shapes
            is_shape &= all([elem >= 0 for elem in user_shape])
            return is_shape


        def generate_buffer(name, dtype, shape):
            if is_shape_tensor(name, dtype):
                buffer = np.array(shape, dtype=dtype)
                G_LOGGER.info("Assuming {:} is a shape tensor. Setting input values to: {:}. If this is not correct, "
                              "please set it correctly in 'input_metadata' or by providing --input-shapes".format(name, buffer), mode=LogMode.ONCE)
            elif np.issubdtype(dtype, np.integer):
                # high is 1 greater than the max int drawn
                buffer = rng.randint(low=self.int_range[0], high=self.int_range[1] + 1, size=shape, dtype=dtype)
            elif np.issubdtype(dtype, np.bool_):
                buffer = rng.randint(low=0, high=2, size=shape).astype(dtype)
            else:
                buffer = (rng.random_sample(size=shape) * (self.float_range[1] - self.float_range[0]) + self.float_range[0]).astype(dtype)

            buffer = np.array(buffer) # To handle scalars, since the above functions return a float if shape is ().
            return buffer


        if self.input_metadata is None and self.user_input_metadata is not None:
            self.input_metadata = self.user_input_metadata

        buffers = OrderedDict()
        for name, (dtype, shape) in self.input_metadata.items():
            if name in self.user_input_metadata:
                user_dtype, user_shape = self.user_input_metadata[name]

                is_valid_shape_override = user_shape is not None and misc.is_valid_shape_override(user_shape, shape)
                if not is_valid_shape_override and not is_shape_tensor(name, dtype):
                    G_LOGGER.warning("Input tensor: {:24} | Cannot use provided custom shape: {:}, since this input has "
                                     "a static shape: {:}".format(name, user_shape, shape), mode=LogMode.ONCE)
                else:
                    shape = misc.default_value(user_shape, shape)

                dtype = misc.default_value(user_dtype, dtype)

            static_shape = get_static_shape(name, shape)
            buffers[name] = generate_buffer(name, dtype, shape=static_shape)

        # Warn about unused metadata
        for name in self.user_input_metadata.keys():
            if name not in self.input_metadata:
                msg = "Input tensor: {:24} | Metadata was provided, but the input does not exist in one or more runners.".format(name)
                close_match = misc.find_in_dict(name, self.input_metadata)
                if close_match:
                    msg += "\nMaybe you meant to set: {:}".format(close_match)
                G_LOGGER.warning(msg)

        return buffers


# Caches data loaded by a DataLoader for use across multiple runners.
class DataLoaderCache(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.cache = [] # List[OrderedDict[str, np.ndarray]]


    def __getitem__(self, iteration):
        """
        Load the specified iteration from the cache if present, or generate using the data loader.

        Args:
            iteration (int): The iteration whose data to retrieve.
        """
        if iteration > len(self.cache):
            raise IndexError()

        # Attempts to match existing input buffers to the requested input_metadata
        def coerce_cached_input(index, name, dtype, shape):
            cached_feed_dict = self.cache[iteration]
            cached_name = misc.find_in_dict(name, cached_feed_dict, index)
            assert cached_name is not None

            if cached_name != name:
                G_LOGGER.warning("Input tensor: {:24} | Cached buffer name ({:}) does not match input name ({:}).".format(
                                    name, cached_name, name))

            buffer = cached_feed_dict[cached_name]

            if dtype != buffer.dtype:
                G_LOGGER.warning("Input tensor: {:24} | Cached buffer dtype ({:}) does not match input dtype ({:}), attempting cast. ".format(
                                    name, buffer.dtype, np.dtype(dtype).name))
                buffer = buffer.astype(dtype)

            if not misc.is_valid_shape_override(buffer.shape, shape):
                G_LOGGER.warning("Input tensor: {:24} | Cached buffer shape ({:}) does not match input shape ({:}), attempting reshape. ".format(
                                    name, buffer.shape, shape))
                buffer = misc.try_match_shape(buffer, shape)

            assert buffer.dtype == dtype and misc.is_valid_shape_override(buffer.shape, shape)
            return buffer


        feed_dict = OrderedDict()

        # Reload from data loader if needed
        data_loader_feed_dict = None

        for index, (name, (dtype, shape)) in enumerate(self.input_metadata.items()):
            try:
                buffer = coerce_cached_input(index, name, dtype, shape)
            except AssertionError:
                G_LOGGER.warning("Could not reuse input: {:} across runners. Attempting to reload "
                                "inputs from the data loader. Note that this will only work if the data loader "
                                "supports random access.".format(name))
                try:
                    if data_loader_feed_dict is None:
                        data_loader_feed_dict = self.data_loader[iteration]
                    buffer = data_loader_feed_dict[name]
                except:
                    G_LOGGER.critical("Could not reload inputs from data loader. Are the runners running the same model? "
                                      "If not, please rewrite the data loader to support random access.")
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
        with contextlib.suppress(AttributeError): self.data_loader.input_metadata = input_metadata
        if not self.cache:
            G_LOGGER.verbose("Loading inputs from data loader")
            self.cache = list(self.data_loader)
            if not self.cache:
                G_LOGGER.warning("Data loader did not yield any input data.")
