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
from polygraphy.tools.util import args as args_util, misc as tool_util
from polygraphy.logger import G_LOGGER, LogMode
from polygraphy.common import constants
from polygraphy.tools.base import Tool
from polygraphy.util import misc
import polygraphy

import argparse
import copy
import sys
import os


# Generate a summary line to add as a comment to the script
def generate_summary(model_file, runners, load_results):
    def join_list(lst):
        new_list = copy.copy(lst)
        if len(new_list) > 1:
            new_list[-1] = "and {:}".format(new_list[-1])
        return ", ".join(new_list)

    summary = ""

    if runners:
        summary += "This script "
        if len(runners) > 1:
            summary += "compares "
        else:
            summary += "runs "
        if model_file:
            summary += "{:} ".format(model_file)

        runner_names = {
            "trt": "TensorRT",
            "trt_legacy": "TensorRT Legacy",
            "tf": "TensorFlow",
            "onnxrt": "ONNX Runtime",
            "onnxtf": "ONNX-TensorFlow Backend",
            "cntk": "CNTK"
        }
        runners = [runner_names[runner] for runner in runners]
        summary += "between " if len(runners) > 1 else "using "
        summary += join_list(runners)

    if load_results:
        summary += "\nIt will check against outputs stored in {:}\n".format(join_list(load_results))

    return summary



def add_tf_runner(script, args):
    script.add_import(imports=["TfRunner"], frm="polygraphy.backend.tf")

    graph_name = tool_util.add_tf_loader(script, args)
    config_name = tool_util.add_tf_config_loader(script, args)

    script.add_import(imports=["SessionFromGraph"], frm="polygraphy.backend.tf")
    loader_name = script.add_loader(Script.invoke("SessionFromGraph", graph_name, config=config_name), "build_tf_session")

    runner_str = Script.invoke("TfRunner", loader_name, timeline_path=args.save_timeline)
    script.add_runner(runner_str)


def add_onnxrt_runner(script, args):
    script.add_import(imports=["OnnxrtRunner"], frm="polygraphy.backend.onnxrt")
    onnx_name = tool_util.add_serialized_onnx_loader(script, args)

    script.add_import(imports=["SessionFromOnnxBytes"], frm="polygraphy.backend.onnxrt")
    loader_name = script.add_loader(Script.invoke("SessionFromOnnxBytes", onnx_name), "build_onnxrt_session")

    script.add_runner(Script.invoke("OnnxrtRunner", loader_name))


def add_onnxtf_runner(script, args):
    script.add_import(imports=["OnnxTfRunner", "OnnxFromPath"], frm="polygraphy.backend.onnx")
    script.add_runner(Script.invoke("OnnxTfRunner", tool_util.add_onnx_loader(script, args, suffix="_onnxtf")))


def add_cntk_runner(script, args):
    script.add_import(imports=["CNTKRunner"], frm="polygraphy.backend.cntk")
    script.add_runner(Script.invoke("CNTKRunner", args.model_file))


def add_trt_runner(script, args, data_loader_name):
    script.add_import(imports=["TrtRunner"], frm="polygraphy.backend.trt")

    if args.model_type == "engine":
        loader_name = tool_util.add_trt_serialized_engine_loader(script, args)
    else:
        script.add_import(imports=["EngineFromNetwork"], frm="polygraphy.backend.trt")
        loader_name = tool_util.add_trt_network_loader(script, args)
        config_loader_name = tool_util.add_trt_config_loader(script, args, data_loader_name=data_loader_name)
        loader_str = Script.invoke("EngineFromNetwork", loader_name, config=config_loader_name)
        loader_name = script.add_loader(loader_str, "build_engine")

    SAVE_ENGINE = "SaveEngine"
    save_engine = Script.invoke(SAVE_ENGINE, loader_name, path=args_util.get(args, "save_engine"))
    if save_engine != Script.invoke(SAVE_ENGINE, loader_name):
        script.add_import(imports=[SAVE_ENGINE], frm="polygraphy.backend.trt")
        loader_name = script.add_loader(save_engine, "save_engine")

    script.add_runner(Script.invoke("TrtRunner", loader_name))


def add_trt_legacy_runner(script, args):
    script.add_import(imports=["TrtLegacyRunner"], frm="polygraphy.backend.trt_legacy")
    G_LOGGER.warning("Legacy TensorRT runner only supports implicit batch TensorFlow/UFF, ONNX, and Caffe models")

    if args.model_type == "onnx":
        script.add_import(imports=["ParseNetworkFromOnnxLegacy"], frm="polygraphy.backend.trt_legacy")
        onnx_loader = tool_util.add_onnx_loader(script, args, disable_outputs=True)
        loader_name = script.add_loader(Script.format_str("ParseNetworkFromOnnxLegacy({:})", onnx_loader), "parse_network_from_onnx_legacy")
    elif args.model_type == "caffe":
        script.add_import(imports=["LoadNetworkFromCaffe"], frm="polygraphy.backend.trt_legacy")
        loader_name = script.add_loader(Script.format_str("LoadNetworkFromCaffe({:}, {:}, {:}, {:})", args.model_file, args.caffe_model,
                                                          args.trt_outputs, args.batch_size), "parse_network_from_caffe")
    else:
        script.add_import(imports=["LoadNetworkFromUff"], frm="polygraphy.backend.trt_legacy")
        if args.model_type == "uff":
            script.add_import(imports=["LoadUffFile"], frm="polygraphy.backend.trt_legacy")
            shapes = {name: shape for name, (_, shape) in args.inputs.items()}
            loader_name = script.add_loader(Script.format_str("LoadUffFile({:}, {:}, {:})", args.model_file, misc.default_value(shapes, {}), args.trt_outputs), "load_uff_file")
        else:
            script.add_import(imports=["ConvertToUff"], frm="polygraphy.backend.trt_legacy")
            loader_name = script.add_loader(Script.format_str("ConvertToUff({:}, save_uff={:}, preprocessor={:})", tool_util.add_tf_loader(script, args), args.save_uff, args.preprocessor), "convert_to_uff")
        loader_name = script.add_loader(Script.format_str("LoadNetworkFromUff({:}, uff_order={:})", loader_name, args.uff_order), "uff_network_loader")


    runner_str = Script.format_str("TrtLegacyRunner({:}, {:}, {:}, fp16={:}, tf32={:}, load_engine={:}, save_engine={:}, layerwise={:}, plugins={:})", loader_name, args.workspace, args.batch_size, args.fp16, args.tf32, args.model_file if args.model_type == "engine" else None, args.save_engine, args_util.get(args, "trt_outputs")==constants.MARK_ALL, args.plugins)
    script.add_runner(runner_str)



def add_comparator(script, args, data_loader_name, cmd_run):
    script.add_import(imports=["Comparator"], frm="polygraphy.comparator")
    script.add_import(imports=["sys"])
    comparator_run = Script.invoke("Comparator.run", script.get_runners(), warm_up=args.warm_up,
                                   data_loader=data_loader_name, use_subprocess=args.use_subprocess)
    script.append_suffix(Script.format_str("\n# Runner Execution\nresults = {:}", Inline(comparator_run)))

    if args.load_results:
        G_LOGGER.verbose("Will load runner results from: {:}".format(args.load_results))
        script.add_import(imports=["misc"], frm="polygraphy.util")
        script.append_suffix(Script.format_str("\n# Load results\nfor load_output in {:}:\n{:}results.update(misc.pickle_load(load_output))", args.load_results, Inline(constants.TAB)))

    if args.save_results:
        G_LOGGER.verbose("Will save runner results to: {:}".format(args.save_results))
        script.add_import(imports=["misc"], frm="polygraphy.util")
        script.append_suffix(Script.format_str("\n# Save results\nmisc.pickle_save({:}, results)", args.save_results))

    top_k = args_util.get(args, "top_k")
    if top_k is not None:
        script.add_import(imports=["PostprocessFunc"], frm="polygraphy.comparator")
        script.append_suffix(Script.format_str("\n# Postprocessing - Apply Top-{:}\nresults = Comparator.postprocess(results, PostprocessFunc.topk_func(k={:}))", top_k, top_k))

    script.append_suffix("\nsuccess = True")

    if len(args.runners) > 1 or args.load_results: # Only do comparisons if there's actually something to compare.
        script.append_suffix("# Accuracy Comparison")

        compare_func_str = Script.invoke_if_nondefault("CompareFunc.basic_compare_func", rtol=args.rtol, atol=args.atol,
                                                       check_shapes=False if args.no_shape_check else None,
                                                       fail_fast=args.fail_fast)
        compare_func = None
        if compare_func_str:
            script.add_import(imports=["CompareFunc"], frm="polygraphy.comparator")
            compare_func = "compare_func"
            script.append_suffix(Script.format_str("{:} = {:}", Inline(compare_func), Inline(compare_func_str)))

        compare_accuracy = Script.invoke("Comparator.compare_accuracy", Inline("results"), compare_func=Inline(compare_func) if compare_func is not None else None,
                                    fail_fast=args.fail_fast)
        script.append_suffix(Script.format_str("success &= bool({:})\n", Inline(compare_accuracy)))
    if args.validate:
        script.append_suffix("# Validation\nsuccess &= Comparator.validate(results)\n")

    if cmd_run is None:
        cmd_run = Inline("' '.join(sys.argv)")
    script.append_suffix(Script.format_str('# Report Results\ncmd_run={cmd}\nif success:\n    G_LOGGER.success("PASSED | Command: {{}}".format(cmd_run))\nelse:\n    G_LOGGER.error("FAILED | Command: {{}}".format(cmd_run))', cmd=cmd_run))
    script.append_suffix("sys.exit(0 if success else 1)")


# Generates a script based on command-line arguments
def build_script(args, cmd_run=None):
    script = Script(summary=generate_summary(args.model_file, args.runners, args.load_results))
    tool_util.add_logger_settings(script, args)

    data_loader_name = tool_util.add_data_loader(script, args)

    for runner_arg in args.runners:
        add_runner_func = {
            "tf": add_tf_runner,
            "onnxrt": add_onnxrt_runner,
            "onnxtf": add_onnxtf_runner,
            "cntk": add_cntk_runner,
            "trt": lambda script, args: add_trt_runner(script, args, data_loader_name),
            "trt_legacy": add_trt_legacy_runner,
        }[runner_arg]
        add_runner_func(script, args)

    add_comparator(script, args, data_loader_name=data_loader_name, cmd_run=cmd_run)
    return str(script)


################################# TOOL #################################


class Run(Tool):
    """
    Run inference and compare results across backends.
    """
    def __init__(self):
        self.name = "run"


    def add_parser_args(self, parser):
        parser.add_argument("--gen", "--gen-script", help="Path to save a generated Python script, that will do exactly "
                            "what `run` would. When this option is enabled, `run` will just save the script and exit. "
                            "Use `-` to print the script to the standard output",
                            type=argparse.FileType("w"), dest="gen_script")
        args_util.add_model_args(parser)
        args_util.add_runner_args(parser)
        args_util.add_comparator_args(parser, top_k=True)
        args_util.add_dataloader_args(parser)
        args_util.add_trt_args(parser, network_api=True)
        args_util.add_trt_legacy_args(parser)
        args_util.add_tf_args(parser)
        args_util.add_onnx_args(parser)
        args_util.add_tf_onnx_args(parser)


    def __call__(self, args):
        misc.log_module_info(polygraphy)

        script = build_script(args)

        if args.gen_script:
            with args.gen_script:
                args.gen_script.write(script)

                path = args.gen_script.name
                # Somehow, piping fools isatty, e.g. `polygraphy run --gen-script - | cat`
                if not args.gen_script.isatty() and path not in ["<stdout>", "<stderr>"]:
                    G_LOGGER.info("Writing script to: {:}".format(path))
                    # Make file executable
                    os.chmod(path, os.stat(path).st_mode | 0o111)
        else:
            exec(script)

        return 0
