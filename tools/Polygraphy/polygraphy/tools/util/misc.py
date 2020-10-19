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
from polygraphy.tools.util.script import Script, Inline
from polygraphy.tools.util import args as args_util
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy.common import constants
from polygraphy.util import misc

import os


################################# SCRIPT HELPERS #################################


def add_logger_settings(script, args):
    # Always required since it is used to print the exit message.
    script.append_preimport("from polygraphy.logger import G_LOGGER")

    logger_settings = []
    verbosity_count = args_util.get(args, "verbose")
    if verbosity_count >= 4:
        logger_settings.append("G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE")
    elif verbosity_count == 3:
        logger_settings.append("G_LOGGER.severity = G_LOGGER.SUPER_VERBOSE")
    elif verbosity_count == 2:
        logger_settings.append("G_LOGGER.severity = G_LOGGER.EXTRA_VERBOSE")
    elif verbosity_count == 1:
        logger_settings.append("G_LOGGER.severity = G_LOGGER.VERBOSE")

    if args_util.get(args, "silent"):
        logger_settings.append("G_LOGGER.severity = G_LOGGER.CRITICAL")

    log_format = misc.default_value(args_util.get(args, "log_format"), [])
    for fmt in args.log_format:
        if fmt == "no-colors":
            logger_settings.append("G_LOGGER.colors = False")
        elif fmt == "timestamp":
            logger_settings.append("G_LOGGER.timestamp = True")
        elif fmt == "line-info":
            logger_settings.append("G_LOGGER.line_info = True")

    for setting in logger_settings:
        script.append_preimport(setting)


def _get_outputs_arg(script, args, name):
    outputs = args_util.get(args, name)
    if outputs == constants.MARK_ALL:
        outputs = Inline("constants.MARK_ALL")
        script.add_import(["constants"], frm="polygraphy.common")
    return outputs


def add_tf_loader(script, args, disable_outputs=None, suffix=None):
    if disable_outputs:
        outputs = None
    else:
        outputs = _get_outputs_arg(script, args, "tf_outputs")

    model_file = args_util.get(args, "model_file")
    model_type = args_util.get(args, "model_type")

    save_pb = args_util.get(args, "save_pb")
    save_tensorboard = args_util.get(args, "save_tensorboard")

    if model_type == "ckpt":
        G_LOGGER.verbose("Loading a TensorFlow checkpoint. Please ensure you are not using the --use-subprocess flag".format(model_file), mode=LogMode.ONCE)
        script.add_import(imports=["GraphFromCkpt"], frm="polygraphy.backend.tf")
        loader_id = "load_ckpt"
        loader_str = Script.invoke("GraphFromCkpt", model_file, args_util.get(args, "ckpt"))
    elif model_type == "keras":
        script.add_import(imports=["GraphFromKeras"], frm="polygraphy.backend.tf")
        loader_id = "load_keras"
        loader_str = Script.invoke("GraphFromKeras", model_file)
    else:
        script.add_import(imports=["GraphFromFrozen"], frm="polygraphy.backend.tf")
        G_LOGGER.verbose("Attempting to load as a frozen graph. If this is not correct, please specify --model-type", mode=LogMode.ONCE)
        loader_id = "load_frozen"
        loader_str = Script.invoke("GraphFromFrozen", model_file)

    loader_name = script.add_loader(loader_str, loader_id, suffix=suffix)

    if args_util.get(args, "freeze_graph"):
        script.add_import(imports=["OptimizeGraph"], frm="polygraphy.backend.tf")
        loader_name = script.add_loader(Script.invoke("OptimizeGraph", loader_name), "optimize_graph", suffix=suffix)
    if args_util.get(args, "tftrt"):
        script.add_import(imports=["UseTfTrt"], frm="polygraphy.backend.tf")
        loader_str = Script.invoke("UseTfTrt", loader_name, max_workspace_size=args_util.get(args, "workspace"), fp16=args_util.get(args, "fp16"), int8=args_util.get(args, "int8"),
                                max_batch_size=args_util.get(args, "batch_size"), is_dynamic_op=args_util.get(args, "dynamic_op"), minimum_segment_size=args_util.get(args, "minimum_segment_size"))
        loader_name = script.add_loader(loader_str, "use_tftrt", suffix=suffix)

    MODIFY_TF = "ModifyGraph"
    modify_tf_str = Script.invoke(MODIFY_TF, loader_name, outputs=outputs)
    if modify_tf_str != Script.invoke(MODIFY_TF, loader_name):
        script.add_import(imports=[MODIFY_TF], frm="polygraphy.backend.tf")
        loader_name = script.add_loader(modify_tf_str, "modify_tf")

    engine_dir = None
    if args_util.get(args, "tftrt"):
        engine_dir = args_util.get(args, "save_engine")

    WRITE_TF = "SaveGraph"
    write_tf_str = Script.invoke(WRITE_TF, loader_name, path=save_pb, tensorboard_dir=save_tensorboard, engine_dir=engine_dir)
    if write_tf_str != Script.invoke(WRITE_TF, loader_name):
        script.add_import(imports=[WRITE_TF], frm="polygraphy.backend.tf")
        loader_name = script.add_loader(write_tf_str, "save_tf")

    return loader_name


def add_tf_config_loader(script, args):
    config_loader_str = Script.invoke_if_nondefault("CreateConfig", gpu_memory_fraction=args_util.get(args, "gpu_memory_fraction"),
                               allow_growth=args_util.get(args, "allow_growth"), use_xla=args_util.get(args, "xla"))
    if config_loader_str is not None:
        script.add_import(imports=["CreateConfig"], frm="polygraphy.backend.tf")
        config_loader_name = script.add_loader(config_loader_str, "create_tf_config")
    else:
        config_loader_name = None
    return config_loader_name



def get_modify_onnx_str(script, args, loader_name, disable_outputs=None):
    if disable_outputs:
        outputs = None
        exclude_outputs = None
    else:
        outputs = _get_outputs_arg(script, args, "onnx_outputs")
        exclude_outputs = args_util.get(args, "onnx_exclude_outputs")

    if hasattr(args, "shape_inference"):
        do_shape_inference = args_util.get(args, "shape_inference")
    else:
        do_shape_inference = None if args_util.get(args, "no_shape_inference") else True

    MODIFY_ONNX = "ModifyOnnx"
    modify_onnx_str = Script.invoke(MODIFY_ONNX, loader_name, do_shape_inference=do_shape_inference,
                                    outputs=outputs, exclude_outputs=exclude_outputs)
    if modify_onnx_str != Script.invoke(MODIFY_ONNX, loader_name):
        script.add_import(imports=[MODIFY_ONNX], frm="polygraphy.backend.onnx")
        return modify_onnx_str
    return None


def add_onnx_loader(script, args, disable_outputs=None, suffix=None):
    if args_util.get(args, "model_type") == "onnx":
        script.add_import(imports=["OnnxFromPath"], frm="polygraphy.backend.onnx")
        loader_str = Script.invoke("OnnxFromPath", args_util.get(args, "model_file"))
        loader_name = script.add_loader(loader_str, "load_onnx", suffix=suffix)
    else:
        G_LOGGER.verbose("Attempting to load as a TensorFlow model, using TF2ONNX to convert to ONNX. "
                       "If this is not correct, please specify --model-type", mode=LogMode.ONCE)
        script.add_import(imports=["OnnxFromTfGraph"], frm="polygraphy.backend.onnx")
        loader_str = Script.invoke("OnnxFromTfGraph", add_tf_loader(script, args, disable_outputs=True, suffix=suffix),
                                opset=args_util.get(args, "opset"), fold_constant=False if args_util.get(args, "no_const_folding") else None)
        loader_name = script.add_loader(loader_str, "export_onnx_from_tf", suffix=suffix)

    modify_onnx_str = get_modify_onnx_str(script, args, loader_name, disable_outputs=disable_outputs)
    if modify_onnx_str is not None:
        loader_name = script.add_loader(modify_onnx_str, "modify_onnx")

    save_onnx = args_util.get(args, "save_onnx")
    SAVE_ONNX = "SaveOnnx"
    save_onnx_str = Script.invoke(SAVE_ONNX, loader_name, path=save_onnx)
    if save_onnx_str != Script.invoke(SAVE_ONNX, loader_name):
        script.add_import(imports=[SAVE_ONNX], frm="polygraphy.backend.onnx")
        loader_name = script.add_loader(save_onnx_str, "save_onnx")

    return loader_name


def add_serialized_onnx_loader(script, args, disable_outputs=None):
    model_file = args_util.get(args, "model_file")

    needs_modify = get_modify_onnx_str(script, args, "check_needs_modify", disable_outputs) is not None
    should_import_raw = args_util.get(args, "model_type") == "onnx" and not needs_modify

    if should_import_raw:
        script.add_import(imports=["BytesFromPath"], frm="polygraphy.backend.common")
        onnx_loader = script.add_loader(Script.invoke("BytesFromPath", model_file), "load_serialized_onnx")
    else:
        script.add_import(imports=["BytesFromOnnx"], frm="polygraphy.backend.onnx")
        onnx_loader = add_onnx_loader(script, args, disable_outputs=disable_outputs)
        onnx_loader = script.add_loader(Script.invoke("BytesFromOnnx", onnx_loader), "serialize_onnx")
    return onnx_loader


# If plugins are present, wrap the provided loader/object with LoadPlugins
def _wrap_if_plugins(script, args, obj_name):
    plugins = args_util.get(args, "plugins")
    if plugins:
        script.add_import(imports=["LoadPlugins"], frm="polygraphy.backend.trt")
        loader_str = Script.invoke("LoadPlugins", obj_name, plugins=plugins)
        obj_name = script.add_loader(loader_str, "load_plugins")
    return obj_name


def add_trt_network_loader(script, args):
    model_file = args_util.get(args, "model_file")
    outputs = _get_outputs_arg(script, args, "trt_outputs")

    if args_util.get(args, "network_api"):
        CREATE_NETWORK_FUNC = Inline("create_network")

        script.add_import(imports=["CreateNetwork"], frm="polygraphy.backend.trt")
        script.add_import(imports=["extend"], frm="polygraphy.common.func")

        script.append_prefix("# Manual TensorRT network creation")
        script.append_prefix("@extend(CreateNetwork())")
        script.append_prefix("def {:}(builder, network):".format(CREATE_NETWORK_FUNC))
        script.append_prefix("{tab}import tensorrt as trt\n".format(tab=constants.TAB))
        script.append_prefix("{tab}# Define your network here. Make sure to mark outputs!".format(tab=constants.TAB))
        net_inputs = args_util.get(args, "inputs")
        if net_inputs:
            for name, (dtype, shape) in net_inputs.items():
                script.append_prefix("{tab}{name} = network.add_input(name='{name}', shape={shape}, dtype=trt.float32) # TODO: Set dtype".format(
                                        name=name, shape=shape, tab=constants.TAB))
        script.append_prefix("{tab}# TODO: network.mark_output(...)\n".format(tab=constants.TAB))
        return CREATE_NETWORK_FUNC


    if args_util.get(args, "ext"):
        script.add_import(imports=["NetworkFromOnnxPath"], frm="polygraphy.backend.trt")
        loader_str = Script.invoke("NetworkFromOnnxPath", _wrap_if_plugins(script, args, model_file), explicit_precision=args_util.get(args, "explicit_precision"))
        loader_name = script.add_loader(loader_str, "parse_network_from_onnx")
    else:
        script.add_import(imports=["NetworkFromOnnxBytes"], frm="polygraphy.backend.trt")
        onnx_loader = add_serialized_onnx_loader(script, args, disable_outputs=True)
        loader_str = Script.invoke("NetworkFromOnnxBytes", _wrap_if_plugins(script, args, onnx_loader), explicit_precision=args_util.get(args, "explicit_precision"))
        loader_name = script.add_loader(loader_str, "parse_network_from_onnx")

    MODIFY_NETWORK = "ModifyNetwork"
    modify_network_str = Script.invoke(MODIFY_NETWORK, loader_name, outputs=outputs, exclude_outputs=args_util.get(args, "trt_exclude_outputs"))
    if modify_network_str != Script.invoke(MODIFY_NETWORK, loader_name):
        script.add_import(imports=[MODIFY_NETWORK], frm="polygraphy.backend.trt")
        loader_name = script.add_loader(modify_network_str, "modify_network")

    return loader_name


def add_trt_config_loader(script, args, data_loader_name):
    profiles = []
    for (min_shape, opt_shape, max_shape) in args_util.get(args, "profiles"):
        profile_str = "Profile()"
        for name in min_shape.keys():
            profile_str += Script.format_str(".add({:}, min={:}, opt={:}, max={:})", name, min_shape[name], opt_shape[name], max_shape[name])
        profiles.append(Inline(profile_str))
    if profiles:
        script.add_import(imports=["Profile"], frm="polygraphy.backend.trt")
        sep = Inline("\n{:}".format(constants.TAB))
        profiles = Script.format_str("[{:}{:}\n]", sep, Inline((",{:}".format(sep)).join(profiles)))
        profile_name = script.add_loader(profiles, "profiles")
    else:
        profile_name = None

    calibrator = None
    if args_util.get(args, "int8"):
        script.add_import(imports=["DataLoader"], frm="polygraphy.comparator")
        script.add_import(imports=["Calibrator"], frm="polygraphy.backend.trt")
        calibrator = Script.invoke("Calibrator", data_loader=Inline(data_loader_name) if data_loader_name else Inline("DataLoader()"),
                                   cache=args_util.get(args, "calibration_cache"))

    config_loader_str = Script.invoke_if_nondefault("CreateTrtConfig", max_workspace_size=args_util.get(args, "workspace"), tf32=args_util.get(args, "tf32"),
                                                    fp16=args_util.get(args, "fp16"), int8=args_util.get(args, "int8"), strict_types=args_util.get(args, "strict_types"),
                                                    profiles=profile_name, calibrator=Inline(calibrator) if calibrator else None)
    if config_loader_str is not None:
        script.add_import(imports=["CreateConfig as CreateTrtConfig"], frm="polygraphy.backend.trt")
        config_loader_name = script.add_loader(config_loader_str, "create_trt_config")
    else:
        config_loader_name = None
    return config_loader_name


def add_trt_serialized_engine_loader(script, args):
    script.add_import(imports=["EngineFromBytes"], frm="polygraphy.backend.trt")
    script.add_import(imports=["BytesFromPath"], frm="polygraphy.backend.common")

    load_engine = script.add_loader(Script.invoke("BytesFromPath", args_util.get(args, "model_file")), "load_engine")
    return script.add_loader(Script.invoke("EngineFromBytes", _wrap_if_plugins(script, args, load_engine)), "deserialize_engine")


def add_data_loader(script, args):
    def omit_none_tuple(tup):
        if all([elem is None for elem in tup]):
            return None
        return tup

    int_range = omit_none_tuple(tup=(args_util.get(args, "int_min"), args_util.get(args, "int_max")))
    float_range = omit_none_tuple(tup=(args_util.get(args, "float_min"), args_util.get(args, "float_max")))

    input_metadata_str = Inline(repr(args_util.get(args, "inputs"))) if args_util.get(args, "inputs") else None
    if input_metadata_str:
        script.add_import(imports=["TensorMetadata"], frm="polygraphy.common")

    data_loader = Script.invoke_if_nondefault("DataLoader", seed=args_util.get(args, "seed"), iterations=args_util.get(args, "iterations"),
                                              input_metadata=input_metadata_str, int_range=int_range, float_range=float_range)
    if data_loader is not None:
        data_loader_name = Inline("data_loader")
        script.add_import(imports=["DataLoader"], frm="polygraphy.comparator")
        script.append_prefix(Script.format_str("\n# Inference Inputs Loader\n{:} = {:}\n", data_loader_name, Inline(data_loader)))
    else:
        data_loader_name = None
    return data_loader_name


################################# PYTHON HELPERS #################################


def get_tf_model_loader(args):
    script = Script()
    loader_name = add_tf_loader(script, args)
    exec(str(script), globals(), locals())
    return locals()[loader_name]


def get_onnx_model_loader(args):
    script = Script()
    loader_name = add_onnx_loader(script, args)
    exec(str(script), globals(), locals())
    return locals()[loader_name]


def get_trt_network_loader(args):
    script = Script()
    loader_name = add_trt_network_loader(script, args)
    exec(str(script), globals(), locals())
    return locals()[loader_name]


def get_trt_config_loader(args, data_loader):
    script = Script()
    loader_name = add_trt_config_loader(script, args, data_loader_name="data_loader")
    exec(str(script), globals(), locals())
    return locals()[loader_name]


def get_trt_serialized_engine_loader(args):
    script = Script()
    loader_name = add_trt_serialized_engine_loader(script, args)
    exec(str(script), globals(), locals())
    return locals()[loader_name]


def get_data_loader(args):
    script = Script()
    data_loader_name = add_data_loader(script, args)
    if data_loader_name is None: # All arguments are default
        from polygraphy.comparator import DataLoader
        return DataLoader()
    exec(str(script), globals(), locals())
    return locals()[data_loader_name]
