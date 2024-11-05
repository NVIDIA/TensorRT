#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Tuple, Union
import numpy as np
from ._export import public_api

# Symbolic expression for a given dimension of a tensor
@public_api()
class ShapeExpr:
    """
    Symbolic expression for single dimension of a tensor
    """
    _exprBuilder = None  # trt.IExprBuilder instance. Populated when a shape-calculation context is entered.

    def __init__(self, value: Union[int, trt.IDimensionExpr, "ShapeExpr"] = None):
        """
        Args:
            value (Union[int, trt.IDimensionExpr, ShapeExpr], optional): Constant or another symbolic expression. Defaults to creating a fake shape expression.
        """
        self._is_dummy = False
        self._dim_expr = None
        self._is_size_tensor = False
        if value is None:
            self._is_dummy = True
        elif isinstance(value, int):
            if self._exprBuilder is None:
                self._dim_expr = None
                self._is_dummy = True
            else:
                self._dim_expr = ShapeExpr._exprBuilder.constant(value)
        elif isinstance(value, trt.IDimensionExpr):
            self._dim_expr = value
        elif isinstance(value, ShapeExpr):
            self._dim_expr = value._dim_expr
            self._is_dummy = value._is_dummy
            self._is_size_tensor = value._is_size_tensor

    def _op(self, op: trt.DimensionOperation, other: Union[int, "ShapeExpr"]):
        if self._is_size_tensor:
            raise ValueError("It is not permitted to perform binary operations on size tensor expressions") # trt limitation
        if self._is_dummy:
            return ShapeExpr()
        if isinstance(other, int):
            other = ShapeExpr(other)
        return ShapeExpr(ShapeExpr._exprBuilder.operation(op, self._expr, other._expr))

    # Binary operations for +, -, *, //, ==. <
    # Those for ceil_div, max and min are provided as top-level functions of tensorrt.plugin
    def __add__(self, other: Union[int, "ShapeExpr"]):
        return self._op(trt.DimensionOperation.SUM, other)

    def __sub__(self, other: Union[int, "ShapeExpr"]):
        return self._op(trt.DimensionOperation.SUB, other)

    def __mul__(self, other: Union[int, "ShapeExpr"]):
        return self._op(trt.DimensionOperation.PROD, other)

    def __floordiv__(self, other: Union[int, "ShapeExpr"]):
        return self._op(trt.DimensionOperation.FLOOR_DIV, other)

    def __eq__(self, other: Union[int, "ShapeExpr"]):
        return self._op(trt.DimensionOperation.EQUAL, other)

    def __lt__(self, other: Union[int, "ShapeExpr"]):
        return self._op(trt.DimensionOperation.LESS, other)

    def __repr__(self):
        if self._is_dummy:
            return f"FakeShapeExpr[id={id(self)}]"
        elif not self.is_constant:
            return f"ShapeExpr[id={id(self)}]"
        return f"ShapeExpr[{self._expr.get_constant_value()}]"

    # A ShapeExpr may be "fake" when it is accessed in a non-shape calculation context. Fake `ShapeExpr`s are externally indistinguishable unless `is_constant` or `constant_value` is required.
    # Therefore, constant checks/access must occur conditionally after evaluating `is_fake`.
    @property
    def is_fake(self) -> bool:
        """
        A ShapeExpr may be "fake" when it is accessed in a non-shape calculation context.
        Fake `ShapeExpr`s are externally indistinguishable unless `is_constant` or `constant_value` is required.
        """
        return self._is_dummy

    @property
    def is_size_tensor(self) -> bool:
        """
        `True` if this represents a size tensor, `False` otherwise.
        """
        return self._is_size_tensor

    @property
    def is_constant(self) -> bool:
        """
        `True` if this shape expression is a build-time constant, `False` otherwise.

        Raises:
            RuntimeError: For fake :class:`ShapeExpr`\s. Check :attr:`is_fake` to determine accessibility.
        """
        if self._is_dummy:
            raise RuntimeError(
                "Not accessible for fake 'ShapeExpr's. Check is_fake to determine accessibility."
            )
        return self._expr.is_constant()

    def constant_value(self) -> int:
        """
        Return value of the constant shape expression.

        Raises:
            RuntimeError: For non-constant shape expressions. Check :attr:`is_constant` to determine accessibility.
        """
        if not self.is_constant:
            raise RuntimeError(
                "Not accessible for non-constant shape expressions. Check is_constant to determine accessibility."
            )
        return self._expr.get_constant_value()
    
    # Evaluate the underlying trt.IDimensionExpr, if so done lazily
    @property
    def _expr(self):
        return self._dim_expr

@public_api()
class SizeTensorShapeExpr(ShapeExpr):
    """
    Extends :class:`ShapeExpr`

    A shape expression that represent a size tensor
        
    """
    def __init__(self, size_tensor_desc: "SizeTensorDesc"):
        """
        .. note:: It is recommended to use :attr:`SizeTensorDesc.expr` to get a :class:`SizeTensorShapeExpr` representing a size tensor
        """
        super().__init__()
        self._is_size_tensor = True
        self._is_dummy = size_tensor_desc.opt.is_fake
        self._size_tensor_desc = size_tensor_desc

    def _op(self, op: trt.DimensionOperation, other: Union[int, "ShapeExpr"]):
        raise ValueError("It is not permitted to perform binary operations on size tensor expressions") # TRT limitation
    
    @property
    def is_constant(self):
        if self._is_dummy:
            raise RuntimeError(
                "Not accessible for fake 'ShapeExpr's. Check is_fake to determine accessibility."
            )
        return False
    
    @property
    def _expr(self):
        if self._dim_expr is not None:
            return self._dim_expr

        self._dim_expr = super()._exprBuilder.declare_size_tensor(self._size_tensor_desc.index, self._size_tensor_desc.opt._expr, self._size_tensor_desc.upper_bound._expr)
        return self._dim_expr
    
    def __repr__(self):
        return f"ShapeExpr[is_size_tensor = True, id={id(self)}]"

# Iterable holding `ShapeExpr`s
@public_api()
class ShapeExprs:
    def __init__(self, length: int, _is_dummy: bool = False):
        """
        Iterable holding :class:`ShapeExpr`\s

        Args:
            length (int): Number of dimensions of the tensor
        """
        self._length = length
        self._is_dummy = _is_dummy
        if _is_dummy:
            self._shapes = [ShapeExpr()] * length
        else:
            self._shapes = [None] * length

    @classmethod
    def from_tuple(cls, shape_exprs: Tuple[Union[ShapeExpr, int]]) -> "ShapeExprs":
        """
        Args:
            shape_exprs (Tuple[Union[ShapeExpr, int]]): Tuple to construct :class:`ShapeExprs` from
        """
        shape_exprs_ = tuple([e if isinstance(e, ShapeExpr) else ShapeExpr(e) for e in shape_exprs])
        inst = cls(len(shape_exprs_))
        inst._shapes = list(shape_exprs_)
        return inst

    def numel(self) -> ShapeExpr:
        """
        Returns a symbolic expression for the number of elements
        """
        ret = ShapeExpr(1)
        for s in self._shapes:
            ret *= s
        return ret

    def __iter__(self):
        return iter(self._shapes)

    def __getitem__(self, index):
        return self._shapes[index]

    def __len__(self):
        return self._length

    def __setitem__(self, index, shape):
        if index >= self._length:
            raise IndexError("Index out of range")
        self._shapes[index] = shape

    def __repr__(self):
        return f"ShapeExprs[{', '.join([s.__repr__() for s in self._shapes])}]"


# Numerical representation of a tensor shape
@public_api()
class Shape:
    """
    Numerical representation of a tensor shape
    """
    def __init__(
        self, tensor_desc: Union[int, trt.DynamicPluginTensorDesc, trt.PluginTensorDesc]
    ):
        self._desc = tensor_desc
        self._is_dynamic = None  # set lazily
        if isinstance(tensor_desc, trt.DynamicPluginTensorDesc):
            self._length = len(tensor_desc.desc.dims)
            self._shapes = tensor_desc.desc.dims
        elif isinstance(tensor_desc, trt.PluginTensorDesc):
            self._length = len(tensor_desc.dims)
            self._shapes = tensor_desc.dims

    def numel(self) -> int:
        """
        Number of elements contained

        Raises:
            ValueError: When :attr:`is_dynamic` is `True`
        """
        if self.is_dynamic:
            raise ValueError("Shape has at least one dynamic dimension.")
        return int(np.prod(self._shapes))

    def __iter__(self):
        yield from self._shapes

    def __getitem__(self, index):
        return self._shapes[index]

    def __len__(self):
        return self._length

    def __str__(self):
        return "Shape" + str(tuple(self))

    @property
    def is_dynamic(self) -> bool:
        """
        `True` if this tensor has at least one dynamic dimension, `False` otherwise.
        """
        if self._is_dynamic is not None:
            return self._is_dynamic

        self._is_dynamic = False
        for d in self._shapes:
            if d == -1:
                self._is_dynamic = True

        return self._is_dynamic

    @property
    def opt(self) -> Tuple[int]:
        """
        Optimum value of dimensions specified for auto-tuning.
        """
        if not self.is_dynamic:
            raise ValueError("opt property is only accessible if is_dynamic is true")
        return tuple(self._desc.opt)

    @property
    def min(self) -> Tuple[int]:
        """
        Lower bounds on tensor's dimensions.
        """
        if not self.is_dynamic:
            raise ValueError("min property is only accessible if is_dynamic is true")
        return tuple(self._desc.min)

    @property
    def max(self) -> Tuple[int]:
        """
        Upper bounds on tensor's dimensions.
        """
        if not self.is_dynamic:
            raise ValueError("max property is only accessible if is_dynamic is true")
        return tuple(self._desc.max)

    def __setitem__(self, index, val):
        if index >= self._length:
            raise IndexError("Index out of range")
        self._shapes.desc[index] = val


# Descriptor for a tensor
# A `TensorDesc` never contains nor refers to any tensor data.
@public_api()
class TensorDesc:
    """
    Descriptor for a tensor
    A `TensorDesc` never contains nor refers to any tensor data.
    """
    def __init__(self, shape_expr: ShapeExprs = None, dtype: trt.DataType = None, format: trt.TensorFormat = None, scale: float = None):
        """
        Args:
            shape_expr (ShapeExprs): The data with which to initialize the tensor.
            dtype (trt.DataType): The data type of the tensor.
            format (trt.TensorFormat): Format (layout) of the tensor.
            scale (float): Scale for INT8 data type.

        .. code-block:: python
            :linenos:
            :caption: Creates a TensorDesc with constant shape expressions

            tensor = trt.TensorDesc((10, 2, 32, 32), dtype=trt.float32)

        .. code-block:: python
            :linenos:
            :caption: Creates a TensorDesc from shape expression of another TensorDesc
            
            tensor = trt.from_shape_expr(other.shape_expr, dtype=trt.float32)
        """

        # `TensorDesc` may or may not have `Shape` information but always has symbolic shape expressions and dtype
        self._shape_expr = shape_expr
        self._dtype = dtype

        # `shape`, `format`, and `scale` are only accessible if `has_shape`. Presently, this would be inside autotune.
        self._shape = None
        self._format = format
        self._scale = scale

        self._aliased_to = None
        self._immutable = False

    def numel(self) -> int:
        """
        Returns:
            Returns an int with the number of elements of the tensor. 

        .. warning::
            Should only be called when TensorDesc.has_shape is true. If a symbolic expression for the number of elements is required, query TensorDesc.shape_expr.numel().
        """
        if not self.has_shape:
            raise ValueError(
                "TensorDesc has no shape information available at this stage. Inspect TensorDesc.has_shape to determine availability."
            )
        return int(np.prod(self.shape))
    
    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self._shape_expr)
    
    @property
    def is_size_tensor(self):
        return False

    # Return a `TensorDesc` that has identical properties to `self` but is mutable
    def like(self) -> "TensorDesc":
        """
        Returns:
            Returns a TensorDesc which has identical properties to this tensor, and is mutable. 

        .. code-block:: python
            :linenos:
            :caption: Communicate that output tensor has identical properties to the input tensor
            
            @tensorrt.plugin.register("my::plugin")
            def _(inp: tensorrt.plugin.TensorDesc) -> tensorrt.plugin.TensorDesc:
                return inp.like()
        """
        cloned = TensorDesc()
        cloned.__dict__.update(self.__dict__)
        cloned._immutable = False
        return cloned

    # Return a `TensorDesc` that has identical properties to `self` AND is aliased to `self` (would result in a `Tensor` during enqueue sharing the same data buffer)
    def aliased(self) -> "TensorDesc":
        """
        Returns:
            Returns a TensorDesc which has identical properties and is aliased to this tensor (would result in a `Tensor` during enqueue sharing the same data buffer).
            Returned TensorDesc is immutable.

        .. code-block:: python
            :linenos:
            :caption: Communicate that output tensor has identical properties to the input tensor
            
            @tensorrt.plugin.register("my::plugin")
            def _(inp: tensorrt.plugin.TensorDesc) -> tensorrt.plugin.TensorDesc:
                return inp.aliased()
        """
        cloned = TensorDesc()
        cloned.__dict__.update(self.__dict__)
        cloned._immutable = False
        cloned._aliased_to = self
        cloned._immutable = True
        return cloned

    def get_aliased(self) -> "TensorDesc":
        """
        Returns:
            Returns a TensorDesc for the tensor which this tensor is aliased to. Returns None is this tensor is not aliased to any other tensor.
        """
        return self._aliased_to

    def _validate_has_shape(self) -> None:
        if not self.has_shape:
            raise ValueError(
                "TensorDesc has no shape information available at this stage. Inspect TensorDesc.has_shape to determine availability."
            )
        
    def _validate_not_immutable(self):
        if hasattr(self, "_immutable") and self._immutable:
            raise ValueError("Cannot modify immutable TensorDesc")

    @property
    def shape_expr(self) -> ShapeExprs:
        """
        Symbolic expressions for the tensor shape. 
        """
        return self._shape_expr

    @property
    def dtype(self) -> trt.DataType:
        """
        Data type of the tensor. 
        """
        return self._dtype
    
    @property
    def shape(self) -> Shape:
        """
        The (concrete) shape of the tensor. 

        .. warning::
            Only accessible when TensorDesc.has_shape is true.
        """
        self._validate_has_shape()
        return self._shape

    @property
    def format(self) -> trt.TensorFormat:
        """
        The format of the tensor. 

        .. warning::
            Only accessible when TensorDesc.has_shape is true.
        """
        self._validate_has_shape()
        return self._format

    @property
    def scale(self) -> float:
        """
        Scale for INT8 data type. 

        .. warning::
            Only accessible when TensorDesc.has_shape is true.
        """
        self._validate_has_shape()
        return self._scale
    
    
    @shape_expr.setter
    def shape_expr(self, value):
        self._shape_expr = value

    @dtype.setter
    def dtype(self, value):
        self._dtype = value
        
    @shape.setter
    def shape(self, value):
        self._validate_not_immutable()
        self._shape = value

    @format.setter
    def format(self, value):
        self._validate_not_immutable()
        self._format = value

    @scale.setter
    def scale(self, value):
        self._validate_not_immutable()
        self._scale = value

    @property
    def is_aliased(self) -> bool:
        """
        True if this tensor is aliased to another tensor, False otherwise.
        """
        return self._aliased_to is not None

    @property
    def has_shape(self) -> bool:
        """
        True if this tensor has concrete shape information, False otherwise.
        """
        return self._shape is not None

    @property
    def is_dynamic(self) -> bool:
        """
        `True` if this tensor has at least one dynamic dimension, `False` otherwise.
        """
        if not self.has_shape:
            raise ValueError(
                "TensorDesc has no shape information available at this stage. Inspect TensorDesc.has_shape to determine availability."
            )
        return self.shape.is_dynamic

    @property
    def has_shape_expr(self) -> bool:
        """
        True if this tensor has symbolic shape expressions, False otherwise.
        """
        return self.shape_expr is not None

    def __setattr__(self, name, value):
        if hasattr(self, "_immutable") and self._immutable and name != "_immutable":
            raise ValueError("Cannot modify immutable TensorDesc properties")
        super().__setattr__(name, value)

@public_api()
class SizeTensorDesc(TensorDesc):
    """
    Extends :class:`TensorDesc`

    Descriptor for a size tensor: a scalar of either INT32 or INT64 data type used to express the extent of a data-dependent dimension.
    """
    def __init__(self, opt: ShapeExpr, upper_bound: ShapeExpr):
        """
        Args:
            opt (ShapeExpr): Symbolic expression for the extent of this size tensor to use in the autotune process of the engine build
            upper_bound (ShapeExpr): Symbolic expression for the upper-bound of this size tensor

        .. note:: It is recommended to construct a size tensor using :func:`size_tensor` instead of using this constructor directly
        """
        super().__init__(ShapeExprs(0), trt.int32)
        self._opt = opt
        self._upper_bound = upper_bound
        self._index = None
        self._expr = SizeTensorShapeExpr(self)
    
    @property
    def is_size_tensor(self):
        return True
    
    @property
    def opt(self) -> ShapeExpr:
        """
        Symbolic expression for the extent of this size tensor to use in the autotune process of the engine build
        """
        return self._opt

    @property
    def upper_bound(self) -> ShapeExpr:
        """
        Symbolic expression for the upper-bound of this size tensor
        """
        return self._upper_bound

    @property
    def index(self) -> int:
        """
        Output index at which this size tensor resides
        """
        return self._index
    
    def _set_index(self, idx: int):
        self._index = idx

    def expr(self) -> SizeTensorShapeExpr:
        """
        Symbolic expression for this size tensor
        """
        return self._expr


# A tensor representation that carries data
@public_api()
class Tensor:
    """
    Representation of a tensor that carries data

    :class:`Tensor` objects are strictly *descriptors* of a tensor with an underlying data buffer. `tensorrt.plugin` does not provide any APIs that perform standard data-altering operations on :class:`Tensor`\s.

    Supports `__cuda_array_interface__` for interoperability with other frameworks.

    """
    def __init__(self):
        self._data_ptr = None
        self._shape = None
        self._format = None
        self._dtype = None
        self._scale = None
        self._strides = None

        self._aliased_to = None
        self._stream = None
        self._read_only = None
        self._immutable = False

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self._shape)
    
    @property
    def data_ptr(self) -> int:
        """
        Pointer to the data buffer of this tensor
        """
        return self._data_ptr
    
    @property
    def dtype(self) -> trt.DataType:
        """
        Data type of the tensor. 
        """
        return self._dtype
    
    @property
    def shape(self) -> Shape:
        """
        The (concrete) shape of the tensor.
        """
        return self._shape

    @property
    def format(self) -> trt.TensorFormat:
        """
        The format of the tensor.
        """
        return self._format

    @property
    def scale(self) -> float:
        """
        Scale for INT8 data type.
        """
        return self._scale
    
    @property
    def strides(self) -> Tuple[int]:
        """
        Strides of this tensor.
        """
        return self._strides

    @data_ptr.setter
    def data_ptr(self, value):
        self._data_ptr = value
    
    @dtype.setter
    def dtype(self, value):
        self._dtype = value
        
    @shape.setter
    def shape(self, value):
        self._shape = value

    @format.setter
    def format(self, value):
        self._format = value

    @scale.setter
    def scale(self, value):
        self._scale = value

    @strides.setter
    def strides(self, value):
        self._strides = value

    def numel(self) -> int:
        """
        Returns the number of elements of the tensor

        Raises:
            ValueError: If the tensor has a data-dependent dimension. Examine :attr:`is_data_dependent` to determine whether the tensor is data-dependent.

        Returns:
            int: Number of elements of the tensor
        """
        if self.is_data_dependent:
            raise ValueError(
                "Tensor has a data-dependent dimension. Examine Tensor.shape to determine wildcards (representing data-dependent dimensions)."
            )
        return int(np.prod(self._shape))

    @property
    def __cuda_array_interface__(self):
        if self._dtype in [trt.DataType.BF16, trt.DataType.FP8, trt.DataType.INT4]:
            raise ValueError(
                f"Handling {self._dtype} via '__cuda_array_interface__' is not supported"
            )

        desc = {
            "shape": tuple(self._shape),
            "typestr": np.dtype(trt.nptype(self._dtype)).str,
        }
        desc["stream"] = self._stream
        desc["version"] = 3
        desc["data"] = (
            self._data_ptr,
            False,
        )  # torch does not support read_only flag. Always set to False -- it is user's responsibility to respect implied read-write restriction(s).
        desc["strides"] = tuple(
            [s * np.dtype(trt.nptype(self._dtype)).itemsize for s in self._strides]
        )

        return desc

    def __setattr__(self, name, value):
        if hasattr(self, "_immutable") and self._immutable and name != "_immutable":
            raise ValueError("Cannot modify immutable Tensor properties")
        super().__setattr__(name, value)

    def get_aliased(self) -> "Tensor":
        """
        Returns:
            Returns :class:`Tensor` of the tensor which this tensor is aliased to. Returns None is this tensor is not aliased to any other tensor.
        """
        return self._aliased_to

    @property
    def is_aliased(self):
        """
        True if this tensor is aliased to another tensor, False otherwise.
        """
        return self._aliased_to is None

    @property
    def is_data_dependent(self):
        """
        True if this tensor contains at least one data-dependent dimension, False otherwise.
        """
        return self._shape.is_dynamic

    # Return a `Tensor` which has the same `data_ptr` as `self` but has the provided shape.
    def aliased(self, shape: Union[Shape, Tuple[int], trt.PluginTensorDesc] = None) -> "Tensor":
        """
        Return a :class:`Tensor` which has the same :attr:`data_ptr` as this but has the provided `shape`.

        Args:
            shape (Union[Shape, Tuple[int], trt.PluginTensorDesc], optional): Required shape of the new tensor (must have the same volume). Defaults to same shape.

        Raises:
            ValueError: If `shape` is not a supported type or if it does not have the same volume
        """
        cloned = Tensor()
        cloned.__dict__.update(self.__dict__)
        cloned._immutable = False
        if isinstance(shape, trt.PluginTensorDesc):
            cloned._shape = Shape(shape)
        elif isinstance(shape, Shape):
            cloned._shape = shape
        elif isinstance(shape, tuple):
            desc = trt.PluginTensorDesc()
            desc.dims = shape
            desc.type = self._dtype
            desc.format = self._format
            desc.scale = self._scale
            cloned._shape = Shape(desc)
        elif shape is None:
            pass
        else:
            raise ValueError("Unsupported type for 'shape'")

        # If either the `shape` or self._shape has a wildcard, we allow aliasing
        if not self.is_data_dependent and cloned.is_data_dependent:
            if cloned._shape.numel() > self.numel():
                raise ValueError("Volume of this tensor is less than the provided 'shape'.")

        cloned._aliased_to = self
        return cloned
