#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tensorrt as trt
import types
import typing
from typing import Callable, Tuple, List
import numpy as np
from ._plugin_class import _TemplateJITPlugin
from ._export import IS_AOT_ENABLED
if IS_AOT_ENABLED:
    from ._plugin_class import _TemplateAOTPlugin
from ._validate import (
    _parse_register_inputs,
    _parse_register_return,
    _validate_autotune,
    _validate_impl,
    _validate_aot_impl,
    _validate_name_and_namespace,
)
from ._utils import (
    _built_in_to_plugin_field_type,
    _join_with,
    _numpy_to_plugin_field_type,
    _is_numpy_array,
    _infer_numpy_type,
)

from ._export import public_api

# Namespace to which plugins are dynamically bound
# A namespace can be thought of as a library of plugins from the same author/common objective
class _PluginNamespace(types.ModuleType):
    def __init__(self, namespace):
        super().__init__("tensorrt.plugin.op." + namespace)
        self._namespace = namespace

    def define(self, name, plugin_def):
        assert not hasattr(self, name)
        setattr(self, name, plugin_def)

    def __getattr__(self, name):
        raise AttributeError(
            f"'{self.__class__.__name__}' object '{self._namespace}' has no attribute '{name}'"
        )

    def __repr__(self):
        return f'_PluginNamespace(namespace="{self._namespace}")'


# `tensorrt.plugin.op` module to which plugin namespaces are dynamically bound
class _Op(types.ModuleType):
    def __init__(self):
        super().__init__("tensorrt.plugin.op")

    def define_or_get(self, namespace):
        if hasattr(self, namespace):
            return getattr(self, namespace)

        ns = _PluginNamespace(namespace)
        setattr(self, namespace, ns)

        return ns

    def __getattr__(self, name):
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


op = _Op()
public_api(symbol="op")(op)

QDP_CREATORS = {}
QDP_REGISTRY = {}

# Contains metadata about a registered plugin and `__call__()`` that allows for a plugin instance to be created
class PluginDef:
    def __init__(self):
        self.plugin_id = None  # includes namespace (format is ns::name)
        self.register_func = None
        self.impl_func = None
        self.aot_impl_func = None
        self.autotune_func = None
        self.autotune_attr_names = None
        self.input_tensor_names = None
        self.input_attrs = None  # map name -> type
        self.impl_attr_names = None
        self.aot_impl_attr_names = None
        self.num_outputs = None
        self.input_arg_schema = None
        self.expects_tactic = None

    def __call__(
        self, *args, **kwargs
    ) -> Tuple[List[trt.ITensor], List[trt.ITensor], trt.IPluginV3]:
        namespace, name = self.plugin_id.split("::")

        input_tensors = []
        schema_chunks = []

        for t in args:
            if not isinstance(t, trt.ITensor):
                raise ValueError(
                    f"Expected trt.ITensor but got input of type {type(t)}"
                )

            schema_chunks.append("ITensor")
            input_tensors.append(t)

        attrs = {}
        for key, value in kwargs.items():
            if key not in self.input_attrs:
                raise ValueError(
                    f"Unexpected attribute {key} provided. Expected one of {self.input_attrs.keys()}."
                )
            attrs[key] = value
            attr_annotation = self.input_attrs[key]
            if isinstance(value, np.ndarray):
                if typing.get_origin(attr_annotation) == np.ndarray:
                    np_dtype = typing.get_args(typing.get_args(attr_annotation)[1])[0]
                    if np.dtype(np_dtype) != np.dtype(value.dtype):
                        raise ValueError(
                            f"Unexpected dtype '{np.dtype(value.dtype)}' for attribute '{key}'. Expected '{np_dtype}'."
                        )
            else:
                if attr_annotation is not type(value):
                    raise ValueError(
                        f"Unexpected type '{type(value)}' for attribute '{key}'. Expected '{attr_annotation}'."
                    )

            schema_chunks.append(key)

        expected_schema = (
            f"({_join_with(['ITensor'] * len(self.input_tensor_names))}"
            + _join_with(self.input_attrs.keys(), True)
            + ")"
        )
        schema = f"({', '.join(schema_chunks)})"

        if schema != expected_schema:
            raise ValueError(
                f"Unexpected schema {schema} received. Expected {expected_schema}."
            )

        if self.plugin_id in QDP_CREATORS:
            plg_creator = trt.get_plugin_registry().get_creator(name, "1", namespace)
        else:
            attrs_types = {}
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    attrs_types[key] = (False, value.dtype)  # (builtin?, type)
                else:
                    attrs_types[key] = (True, type(value))  # (builtin?, type)

            plg_creator = _register_plugin_creator(name, namespace, attrs_types)

        fields = []
        for key, value in attrs.items():
            if isinstance(value, np.ndarray):
                np_type = np.dtype(value.dtype)
                if np_type == np.float16:
                    fields.append(
                        trt.PluginField(
                            key, value.tobytes(), trt.PluginFieldType.UNKNOWN
                        )
                    )
                else:
                    fields.append(
                        trt.PluginField(
                            key, value, _numpy_to_plugin_field_type[np_type]
                        )
                    )
            elif isinstance(value, str):
                fields.append(
                    trt.PluginField(key, value.encode(), trt.PluginFieldType.CHAR)
                )
            elif isinstance(value, bytes):
                fields.append(trt.PluginField(key, value, trt.PluginFieldType.UNKNOWN))
            else:
                fields.append(
                    trt.PluginField(
                        key,
                        np.array([value]),
                        _built_in_to_plugin_field_type[type(value)],
                    )
                )

        def create_plugin_instance(quick_plugin_creation_request: "trt.QuickPluginCreationRequest" = None):
            if quick_plugin_creation_request is None:
                plg = plg_creator.create_plugin(
                    name,
                    namespace,
                    trt.PluginFieldCollection(fields),
                    trt.TensorRTPhase.BUILD
                )
            else:
                plg = plg_creator.create_plugin(
                    name,
                    namespace,
                    trt.PluginFieldCollection(fields),
                    trt.TensorRTPhase.BUILD,
                    quick_plugin_creation_request
                )

            return input_tensors, [], plg

        return create_plugin_instance

class _TemplatePluginCreator(trt.IPluginCreatorV3Quick):
    def __init__(self, name, namespace, attrs):
        trt.IPluginCreatorV3Quick.__init__(self)
        self.name = name
        self.plugin_namespace = namespace
        self.plugin_version = "1"
        field_names = []
        for name, (builtin, type_) in attrs.items():
            if builtin:
                if type_ is str:
                    field_names.append(
                        trt.PluginField(name, b"", trt.PluginFieldType.CHAR)
                    )
                elif type_ is bytes:
                    field_names.append(
                        trt.PluginField(name, b"", trt.PluginFieldType.UNKNOWN)
                    )
                else:
                    field_names.append(
                        trt.PluginField(
                            name, np.array([]), _built_in_to_plugin_field_type[type_]
                        )
                    )
            else:
                field_names.append(
                    trt.PluginField(
                        name, np.array([]), _numpy_to_plugin_field_type[np.dtype(type_)]
                    )
                )

        self.field_names = trt.PluginFieldCollection(field_names)

    def create_plugin(self, name, namespace, fc, phase, qpcr: "trt.QuickPluginCreationRequest" = None):
        desc = QDP_REGISTRY[f"{namespace}::{name}"]
        name = name
        namespace = namespace

        attrs = {}
        for f in fc:
            if f.name not in desc.input_attrs:
                raise AssertionError(
                    f"Unexpected attribute {f.name} provided to create_plugin. Expected one of {desc.input_attrs.keys()}."
                )

            attr_type_annot = desc.input_attrs[f.name]
            if _is_numpy_array(attr_type_annot):
                np_type = _infer_numpy_type(attr_type_annot)
                if np_type == np.float16:
                    attrs[f.name] = np.frombuffer(f.data.tobytes(), dtype=np.float16)
                else:
                    attrs[f.name] = f.data.astype(np_type)
            else:
                if issubclass(attr_type_annot, str):
                    attrs[f.name] = f.data.tobytes().decode("utf-8")
                else:
                    attrs[f.name] = attr_type_annot(f.data)

        jit_or_aot = None # True if JIT is to be created, False if AOT. Not None will be asserted before plugin creation.

        if qpcr is None:
            plg = _TemplateJITPlugin(name, namespace, desc.num_outputs)

            plg.init(
                desc.register_func,
                attrs,
                desc.impl_attr_names,
                desc.impl_func,
                desc.autotune_attr_names,
                desc.autotune_func,
                desc.expects_tactic,
            )

            return plg

        # If there is a strict preference, that takes precedence
        if qpcr == trt.QuickPluginCreationRequest.STRICT_AOT:
            if desc.aot_impl_func is None:
                raise ValueError(f"AOT implementation requested, but not defined for '{desc.plugin_id}'. Was @trt.plugin.aot_impl defined?")
            jit_or_aot = False
        elif qpcr == trt.QuickPluginCreationRequest.STRICT_JIT:
            if desc.impl_func is None:
                raise ValueError(f"JIT implementation requested, but not defined for '{desc.plugin_id}'. Was @trt.plugin.impl defined?")
            jit_or_aot = True
        else:
            aot_defined = desc.aot_impl_func is not None
            jit_defined = desc.impl_func is not None

            # A preferemce must be indicated if both AOT and JIT implementations are defined
            if aot_defined and jit_defined:
                if qpcr == trt.QuickPluginCreationRequest.PREFER_AOT:
                    jit_or_aot = False
                elif qpcr == trt.QuickPluginCreationRequest.PREFER_JIT:
                    jit_or_aot = True
                else:
                    raise ValueError(f"Plugin '{desc.plugin_id}' has both AOT and JIT implementations. NetworkDefinitionCreationFlag.PREFER_AOT_PYTHON_PLUGINS or NetworkDefinitionCreationFlag.PREFER_JIT_PYTHON_PLUGINS should be specified.")
            else:
                # If only one implementation is defined, use that.
                # Any preference specified is ignored. If the preference is strong, a strict flag should have been specified.
                if aot_defined:
                    jit_or_aot = False
                elif jit_defined:
                    jit_or_aot = True
                else:
                    raise ValueError(f"Plugin '{desc.plugin_id}' does not have either a AOT or JIT implementation.")

        assert jit_or_aot is not None

        if jit_or_aot:
            plg = _TemplateJITPlugin(name, namespace, desc.num_outputs)

            plg.init(
                desc.register_func,
                attrs,
                desc.impl_attr_names,
                desc.impl_func,
                desc.autotune_attr_names,
                desc.autotune_func,
                desc.expects_tactic,
            )

        else:
            plg = _TemplateAOTPlugin(name, namespace, desc.num_outputs)

            plg.init(
                desc.register_func,
                attrs,
                desc.aot_impl_attr_names,
                desc.aot_impl_func,
                desc.autotune_attr_names,
                desc.autotune_func
            )

        # the caller can determine if the created plugin is an AOT or JIT plugin by inspecting the interface info
        return plg

def _register_plugin_creator(name: str, namespace: str, attrs_types):
    plg_registry = trt.get_plugin_registry()
    plg_creator = _TemplatePluginCreator(name, namespace, attrs_types)
    plg_registry.register_creator(plg_creator, namespace)
    plg_creator = plg_registry.get_creator(name, "1", namespace)
    QDP_CREATORS[f"{namespace}::{name}"] = plg_creator
    return plg_creator


# Decorator for `tensorrt.plugin.register`
# By default, the plugin will be immediately registered in the TRT plugin registry
# During plugin development/when building engine, lazy registration may be used to delay plugin registration until the plugin is explicitly instantiated using `trt.plugin.op.ns.plugin_name(...)`
@public_api()
def register(plugin_id: str, lazy_register: bool = False) -> Callable:
    """
    Wraps a function to register and describe a TensorRT plugin's IO characteristics. In addition, a complete plugin at least needs an `trt.plugin.impl` function to be registered.

    This API is only intended to be used as a decorator. The decorated function must have type hints for all inputs as well as return value.

    .. code-block:: text

        (inp0: TensorDesc, inp1: TensorDesc, ..., attr0: SupportedAttrType, attr1: SupportedAttrType, ...) -> Union[TensorDesc, Tuple[TensorDesc]]

    * Input tensors are declared first, each described by a tensor descriptor TensorDesc.
    * Plugin attributes are declared next. "SupportedAttrType" must be one of:
       * Supported built-in types: int, float, str, bool, bytes (Note: Lists/tuples of these types are not supported)
       * 1-D Numpy arrays of the following types: int8, int16, int32, int64, float16, float32, float64, bool. These must be annotated with 'numpy.typing.NDArray[dtype]', where 'dtype' is the expected numpy dtype.
    * If the plugin has only one output, the return annotation could be TensorDesc. Tuple[TensorDesc] could be used for any number of outputs.

    By default, the plugin will be immediately registered in the TRT plugin registry. Use the lazy_register argument to change this.

    Args:
        plugin_id: An ID for the plugin in the form "{namespace}::{name}",
            e.g. "my_project::add_plugin". The namespace is used to avoid collisions
            so using your product/project name is recommended.

        lazy_register: During plugin development/when building engine, lazy registration may be used to delay plugin registration until the plugin is explicitly instantiated using `trt.plugin.op.ns.plugin_name(...)`

    .. code-block:: python
        :linenos:
        :caption: Registration of an elementwise plugin (output has same characteristics as the input)

        import tensorrt.plugin as trtp

        @trtp.register("my::add_plugin")
        def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> Tuple[trtp.TensorDesc]:
            return inp0.like()

    """

    def decorator(register_func: Callable):

        plugin_ns, plugin_name = plugin_id.split("::")
        _validate_name_and_namespace(plugin_ns, plugin_name)

        op_namespace = op.define_or_get(plugin_ns)

        if hasattr(op_namespace, plugin_name):
            raise ValueError(
                f"'{op.__class__.__name__}' already has a defintion for '{plugin_name}'"
            )

        (
            tensor_names,
            input_attrs,
            input_arg_schema,
            attrs_types,
        ) = _parse_register_inputs(register_func, lazy_register)

        plugin_def = PluginDef()
        plugin_def.plugin_id = plugin_id
        plugin_def.register_func = register_func
        plugin_def.input_tensor_names = tensor_names
        plugin_def.input_attrs = input_attrs
        plugin_def.input_arg_schema = input_arg_schema

        num_outputs = _parse_register_return(register_func)

        plugin_def.num_outputs = num_outputs
        QDP_REGISTRY[plugin_id] = plugin_def

        if not lazy_register:
            _register_plugin_creator(plugin_name, plugin_ns, attrs_types)

        op_namespace.define(plugin_name, plugin_def)

        return register_func

    return decorator


# Decorator for `tensorrt.plugin.impl`
@public_api()
def impl(plugin_id: str) -> Callable:
    """
    Wraps a function to define an implementation for a plugin already registered through `trt.plugin.register`.

    This API is only intended to be used as a decorator. The decorated function is not required to have type hints for input arguments or return value;
    however, any type hints specified will be validated against the `trt.plugin.register` signature for consistency.

    The schema for the function is as follows:

    .. code-block:: text

        (inp0: Tensor, inp1: Tensor, ..., attr0: SupportedAttrType, attr1: SupportedAttrType, outputs: Tuple[Tensor], stream: int, tactic: Optional[int]) -> None

    * Input tensors are passed first, each described by a `Tensor`.
    * Plugin attributes are declared next.
       * Not all attributes included in `trt.plugin.register` must be specified here -- they could be a subset.
       * Included attributes will be serialized to the TRT engine. Therefore, only attributes the plugin actually needs to perform inference (within the body of `trt.plugin.impl`) should be included.
    * `tactic` is an optional argument. If the plugin is using custom tactics, it must be specified to receive the tactic value to use for the current execution of the plugin.

    Args:
        plugin_id: The ID for the plugin in the form "{namespace}::{name}", which must match that used during `trt.plugin.register`

    .. code-block:: python
        :linenos:
        :caption: Implementation of an elementwise plugin with an OpenAI Triton kernel

        import tensorrt.plugin as trtp
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(y_ptr + offsets, x + 1, mask=mask)

        @trtp.register("my::add_plugin")
        def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> Tuple[trtp.TensorDesc]:
            return inp0.like()

        @trtp.impl("my::add_plugin")
        def add_plugin_impl(inp0: trtp.Tensor, block_size: int, outputs: Tuple[trtp.Tensor], stream: int) -> None:

            n = inp0.numel()
            inp0_t = torch.as_tensor(inp0, device="cuda")
            out_t = torch.as_tensor(outputs[0], device="cuda")

            add_kernel[(triton.cdiv(n, block_size),)](inp0_t, out_t, n, BLOCK_SIZE = block_size)
    """

    def decorator(impl_func: Callable):
        if plugin_id not in QDP_REGISTRY:
            raise ValueError(
                f"Plugin {plugin_id} is not registered. Did you register it with tensorrt.plugin.register API?"
            )

        plugin_def = QDP_REGISTRY[plugin_id]
        impl_attr_names, found_tactic = _validate_impl(impl_func, plugin_def)

        plugin_def.impl_func = impl_func
        plugin_def.impl_attr_names = impl_attr_names
        plugin_def.expects_tactic = found_tactic
        return impl_func

    return decorator

# Decorator for `tensorrt.plugin.aot_impl`
@public_api()
def aot_impl(plugin_id: str) -> Callable:
    """
    Wraps a function to define an Ahead-of-Time (AOT) implementation for a plugin already registered through `trt.plugin.register`.

    This API is only intended to be used as a decorator. The decorated function is not required to have type hints for input arguments or return value;
    however, any type hints specified will be validated against the `trt.plugin.register` signature for consistency.

    The schema for the function is as follows:
    .. code-block:: text

        (inp0: TensorDesc, inp1: TensorDesc, ..., attr0: SupportedAttrType, attr1: SupportedAttrType, outputs: Tuple[TensorDesc], tactic: Optional[int]) -> Tuple[str, str, KernelLaunchParams, SymExprs]

    * Input tensors are passed first, each described by a `TensorDesc`.
    * Plugin attributes are declared next.
       * Not all attributes included in `trt.plugin.register` must be specified here -- they could be a subset.
       * NOTE: Plugin attributes are not serialized into the engine when using an AOT implementation.
    * `tactic` is an optional argument. If the plugin is using custom tactics, it must be specified to receive the tactic value to use for the current execution of the plugin.

    Args:
        plugin_id: The ID for the plugin in the form "{namespace}::{name}", which must match that used during `trt.plugin.register`

    :returns:
        - kernel_name: The name of the kernel.
        - compiled_kernel: Compiled form of the kernel. Presently, only PTX is supported.
        - launch_params: The launch parameters for the kernel
        - extra_args: Symbolic expressions for scalar inputs to the kernel, located after the tensor inputs and before the tensor outputs

    .. code-block:: python
        :linenos:
        :caption: Implementation of an elementwise plugin with an OpenAI Triton kernel

        import tensorrt.plugin as trtp
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, n_elements, y_ptr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(y_ptr + offsets, x + 1, mask=mask)

        @trtp.register("my::add_plugin")
        def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> Tuple[trtp.TensorDesc]:
            return inp0.like()

        @trtp.aot_impl("my::elemwise_add_plugin")
        def add_plugin_aot_impl(
            inp0: trtp.TensorDesc, block_size: int, single_tactic: bool, outputs: Tuple[trtp.TensorDesc], tactic: int
        ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:

            type_str = "fp32" if inp0.dtype == trt.float32 else "fp16"

            src = triton.compiler.ASTSource(
                fn=add_kernel,
                signature=f"*{type_str},i32,*{type_str}",
                constants={
                    "BLOCK_SIZE": block_size,
                },
            )

            compiled_kernel = triton.compile(src)

            N = inp0.shape_expr.numel()
            launch_params = trtp.KernelLaunchParams()

            # grid dims
            launch_params.grid_x = trtp.cdiv(N, block_size)
            # block dims
            launch_params.block_x = compiled_kernel.metadata.num_warps * 32
            # shared memory
            launch_params.shared_mem = compiled_kernel.metadata.shared

            extra_args = trtp.SymIntExprs(1)
            extra_args[0] = trtp.SymInt32(N)

            return compiled_kernel.metadata.name, compiled_kernel.asm["ptx"], launch_params, extra_args
    """
    def decorator(aot_impl_func: Callable):
        if plugin_id not in QDP_REGISTRY:
            raise ValueError(
                f"Plugin {plugin_id} is not registered. Did you register it with tensorrt.plugin.register API?"
            )

        plugin_def = QDP_REGISTRY[plugin_id]
        aot_impl_attr_names = _validate_aot_impl(aot_impl_func, plugin_def)

        plugin_def.aot_impl_func = aot_impl_func
        plugin_def.aot_impl_attr_names = aot_impl_attr_names
        return aot_impl_func

    return decorator


# Decorator for `tensorrt.plugin.autotune`
@public_api()
def autotune(plugin_id: str) -> Callable:
    """
    Wraps a function to define autotune logic for a plugin already registered through `trt.plugin.register`.

    Autotuning is the process by which TensorRT executes the plugin over IO type/format combinations, and any custom tactics advertised as being supported by the plugin.
    The (type, format, tactic) combination with the lowest latency is used to execute the plugin once the engine is built.

    .. note:: An autotune function is optional. If not specified, TensorRT will assume the plugin only supports input types specified at network creation, output types specifeid through `trt.plugin.register`, and linear formats for all I/O.

    This API is only intended to be used as a decorator. The decorated function is not required to have type hints for input arguments or return value; however, any type hints specified will be validated against the `trt.plugin.register` signature for consistency.

    The schema for the function is as follows:

    .. code-block:: text

        (inp0: TensorDesc, inp1: TensorDesc, ..., attr0: SupportedAttrType, attr1: SupportedAttrType, outputs: Tuple[TensorDesc]) -> List[AutoTuneCombination]

    * Input tensors are passed first, each described by a :class:`TensorDesc`.
    * Plugin attributes are declared next. Not all attributes included in `trt.plugin.register` must be specified here -- they could be a subset.
    * The function should return a list of :class:`AutoTuneCombination`\s.

    Args:
        plugin_id: The ID for the plugin in the form "{namespace}::{name}", which must match that used during `trt.plugin.register`

    .. code-block:: python
        :linenos:
        :caption: An elementwise add plugin which supports both FP32 and FP16 linear I/O and wants to be tuned over 2 custom tactics.

        import tensorrt.plugin as trtp

        @trtp.register("my::add_plugin")
        def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> Tuple[trtp.TensorDesc]:
            return inp0.like()

        @trtp.autotune("my::add_plugin")
        def add_plugin_autotune(inp0: trtp.TensorDesc, block_size: int, outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:

            return [trtp.AutoTuneCombination("FP32|FP16, FP32|FP16", "LINEAR", [1, 2])]

    .. code-block:: python
        :linenos:
        :caption: Same as above example but using index-by-index construction of an `AutoTuneCombination`

        import tensorrt.plugin as trtp

        @trtp.register("my::add_plugin")
        def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> Tuple[trtp.TensorDesc]:
            return inp0.like()

        @trtp.autotune("my::add_plugin")
        def add_plugin_autotune(inp0: trtp.TensorDesc, block_size: int, outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
            c = trtp.AutoTuneCombination()
            c.pos(0, "FP32|FP16", "LINEAR")
            c.pos(1, "FP32|FP16") # index 1 is the output. Omitting format is the same as declaring it to be LINEAR.
            c.tactics([1, 2])
            return [c]
    """

    def decorator(autotune_func: Callable):
        if plugin_id not in QDP_REGISTRY:
            raise ValueError(
                f"Plugin {plugin_id} is not registered. Did you register it with tensorrt.plugin.register API?"
            )

        plugin_def = QDP_REGISTRY[plugin_id]
        autotune_attr_names = _validate_autotune(autotune_func, plugin_def)

        plugin_def.autotune_func = autotune_func
        plugin_def.autotune_attr_names = autotune_attr_names

        return autotune_func

    return decorator
