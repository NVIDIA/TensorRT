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
import copy

from polygraphy.common import TensorMetadata, constants
from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.tools.util.script import Inline
from polygraphy.util import misc


def get(args, attr):
    """
    Gets a command-line argument if it exists, otherwise returns None.

    Args:
        args: The command-line arguments.
        attr (str): The name of the command-line argument.
    """
    if hasattr(args, attr):
        return getattr(args, attr)
    return None


def get_outputs(args, name):
    outputs = get(args, name)
    if outputs is not None and len(outputs) == 2 and outputs == ["mark", "all"]:
        outputs = constants.MARK_ALL
    return outputs


def get_outputs_for_script(script, outputs):
    if outputs == constants.MARK_ALL:
        outputs = Inline("constants.MARK_ALL")
        script.add_import(["constants"], frm="polygraphy.common")
    return outputs


def parse_meta(meta_args, includes_shape=True, includes_dtype=True):
    """
    Parses a list of tensor metadata arguments of the form "<name>,<shape>,<dtype>"
    `shape` and `dtype` are optional, but `dtype` must always come after `shape` if they are both enabled.

    Args:
        meta_args (List[str]): A list of tensor metadata arguments from the command-line.
        includes_shape (bool): Whether the arguments include shape information.
        includes_dtype (bool): Whether the arguments include dtype information.

    Returns:
        TensorMetadata: The parsed tensor metadata.
    """
    SEP = ","
    SHAPE_SEP = "x"
    meta = TensorMetadata()
    for orig_tensor_meta_arg in meta_args:
        tensor_meta_arg = orig_tensor_meta_arg

        def pop_meta(name):
            nonlocal tensor_meta_arg
            tensor_meta_arg, _, val = tensor_meta_arg.rpartition(SEP)
            if not tensor_meta_arg:
                G_LOGGER.critical("Could not parse {:} from argument: {:}. Is it separated by a comma "
                                    "(,) from the tensor name?".format(name, orig_tensor_meta_arg))
            if val.lower() == "auto":
                val = None
            return val


        def parse_dtype(dtype):
            if dtype is not None:
                if dtype not in misc.NP_TYPE_FROM_STR:
                    G_LOGGER.critical("Could not understand data type: {:}. Please use one of: {:} or `auto`"
                            .format(dtype, list(misc.NP_TYPE_FROM_STR.keys())))
                dtype = misc.NP_TYPE_FROM_STR[dtype]
            return dtype


        def parse_shape(shape):
            if shape is not None:
                def parse_shape_dim(buf):
                    try:
                        buf = int(buf)
                    except:
                        pass
                    return buf


                parsed_shape = []
                # Allow for quoted strings in shape dimensions
                in_quotes = False
                buf = ""
                for char in shape.lower():
                    if char in ["\"", "'"]:
                        in_quotes = not in_quotes
                    elif not in_quotes and char == SHAPE_SEP:
                        parsed_shape.append(parse_shape_dim(buf))
                        buf = ""
                    else:
                        buf += char
                # For the last dimension
                parsed_shape.append(parse_shape_dim(buf))
                shape = tuple(parsed_shape)
            return shape


        name = None
        dtype = None
        shape = None

        if includes_dtype:
            dtype = parse_dtype(pop_meta("data type"))

        if includes_shape:
            shape = parse_shape(pop_meta("shape"))

        name = tensor_meta_arg
        meta.add(name, dtype, shape)
    return meta


def parse_profile_shapes(default_shapes, min_args, opt_args, max_args):
    """
    Parses TensorRT profile options from command-line arguments.

    Args:
        default_shapes (TensorMetadata): The inference input shapes.

    Returns:
     List[Tuple[OrderedDict[str, Shape]]]:
            A list of profiles with each profile comprised of three dictionaries
            (min, opt, max) mapping input names to shapes.
    """
    def get_shapes(lst, idx):
        nonlocal default_shapes
        default_shapes = copy.copy(default_shapes)
        if idx < len(lst):
            default_shapes.update(parse_meta(lst[idx], includes_dtype=False))

        # Don't care about dtype, and need to override dynamic dimensions
        shapes = {name: misc.override_dynamic_shape(shape) for name, (_, shape) in default_shapes.items()}

        for name, shape in shapes.items():
            if tuple(shapes[name]) != tuple(shape):
                G_LOGGER.warning("Input tensor: {:} | For TensorRT profile, overriding shape: {:} to: {:}".format(name, shape, shapes[name]), mode=LogMode.ONCE)

        return shapes


    num_profiles = max(len(min_args), len(opt_args), len(max_args))

    # For cases where input shapes are provided, we have to generate a profile
    if not num_profiles and default_shapes:
        num_profiles = 1

    profiles = []
    for idx in range(num_profiles):
        min_shapes = get_shapes(min_args, idx)
        opt_shapes = get_shapes(opt_args, idx)
        max_shapes = get_shapes(max_args, idx)
        if sorted(min_shapes.keys()) != sorted(opt_shapes.keys()):
            G_LOGGER.critical("Mismatch in input names between minimum shapes ({:}) and optimum shapes "
                            "({:})".format(list(min_shapes.keys()), list(opt_shapes.keys())))
        elif sorted(opt_shapes.keys()) != sorted(max_shapes.keys()):
            G_LOGGER.critical("Mismatch in input names between optimum shapes ({:}) and maximum shapes "
                            "({:})".format(list(opt_shapes.keys()), list(max_shapes.keys())))

        profiles.append((min_shapes, opt_shapes, max_shapes))
    return profiles
