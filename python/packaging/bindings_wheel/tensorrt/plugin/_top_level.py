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

from typing import Union, Tuple
import tensorrt as trt
from ._tensor import ShapeExpr, TensorDesc, ShapeExprs, SizeTensorDesc
from ._export import public_api

# Miscellaneous top-level functions accessible through `tensorrt.plugin`

# Performs `trt.DimensionOperation.CEIL_DIV`
@public_api()
def cdiv(first: Union[int, ShapeExpr], second: Union[int, ShapeExpr]) -> ShapeExpr:
    """
    Computes symbolic ceiling division of `first` by `second`

    Args:
        first (Union[int, ShapeExpr]): Dividend
        second (Union[int, ShapeExpr]): Divisor

    Raises:
        ValueError: If both arguments are `int`\s or if `second` evaluates to 0

    Returns:
        ShapeExpr: Symbolic expression for the ceiling division of `first` by `second`
    """
    if isinstance(first, int):
        if isinstance(second, int):
            raise ValueError("Both arguments cannot be 'int's")
        first = ShapeExpr(first)

    return first._op(trt.DimensionOperation.CEIL_DIV, second)


# Performs `trt.DimensionOperation.MAX`
@public_api()
def max(first: Union[int, ShapeExpr], second: Union[int, ShapeExpr]) -> ShapeExpr:
    """
    Computes the maximum of `first` and `second`

    Args:
        first (Union[int, ShapeExpr]): First operand
        second (Union[int, ShapeExpr]): Second operand

    Raises:
        ValueError: If both arguments are `int`\s

    Returns:
        ShapeExpr: Symbolic expression for the maximum of `first` and `second`
    """
    if isinstance(first, int):
        if isinstance(second, int):
            raise ValueError("Both arguments cannot be 'int's")
        first = ShapeExpr(first)

    return first._op(trt.DimensionOperation.MAX, second)


# Performs `trt.DimensionOperation.MIN`
@public_api()
def min(first: Union[int, ShapeExpr], second: Union[int, ShapeExpr]) -> ShapeExpr:
    """
    Computes the minimum of `first` and `second`

    Args:
        first (Union[int, ShapeExpr]): First operand
        second (Union[int, ShapeExpr]): Second operand

    Raises:
        ValueError: If both arguments are `int`\s

    Returns:
        ShapeExpr: Symbolic expression for the minimum of `first` and `second`
    """
    if isinstance(first, int):
        if isinstance(second, int):
            raise ValueError("Both arguments cannot be 'int's")
        first = ShapeExpr(first)

    return first._op(trt.DimensionOperation.MIN, second)


# Declare a size tensor descriptor with the specified autotune shape expression `opt` and `upper-bound` shape expression
@public_api()
def size_tensor(opt: ShapeExpr, upper_bound: ShapeExpr) -> SizeTensorDesc:
    """
    Constructs a size tensor with the specified autotune shape expression `opt` and `upper_bound`

    Args:
        opt (ShapeExpr): Symbolic expression for the extent of this size tensor to use in the autotune process of the engine build
        upper_bound (ShapeExpr): Symbolic expression for the upper-bound of this size tensor

    Returns:
        SizeTensorDesc: A tensor descriptor for a size tensor with the specified autotune extent and upper-bound
    """
    return SizeTensorDesc(opt, upper_bound)

# Create a TensorDesc using shape expressions and a dtype
@public_api()
def from_shape_expr(shape_expr: Union[Tuple[Union[ShapeExpr, int]], ShapeExprs], dtype: trt.DataType) -> TensorDesc:
    """
    Constructs a tensor descriptor with the specified shape expression and data type

    Args:
        shape_expr (Union[Tuple[Union[ShapeExpr, int]], ShapeExprs]): Expressions or constants denoting the shape of the tensor
        dtype (trt.DataType): Data type of the tensor

    Returns:
        TensorDesc: Tensor descriptor with the specified shape expression and data type
    """
    if isinstance(shape_expr, tuple):
        shape_expr_ = ShapeExprs.from_tuple(shape_expr)
    else:
        shape_expr_ = shape_expr
    
    return TensorDesc(shape_expr_, dtype)

    
