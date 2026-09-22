# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize supported TensorRT engine-inspector layer payloads."""

from __future__ import annotations

from typing import Any


def extract_layers(payload: Any) -> list[Any]:
    """Return layers from a supported TensorRT engine-inspector payload."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("Layers"), list):
        return payload["Layers"]
    raise ValueError("Layer-info JSON must be a layer array or a TensorRT 11 object containing `Layers`.")


def extract_io_tensors(payload: Any) -> list[Any]:
    """Return optional TensorRT 11 engine I/O descriptors."""
    if not isinstance(payload, dict) or "I/O Tensors" not in payload:
        return []
    io_tensors = payload["I/O Tensors"]
    if not isinstance(io_tensors, list):
        raise ValueError("TensorRT 11 `I/O Tensors` must be an array when present.")
    return io_tensors
