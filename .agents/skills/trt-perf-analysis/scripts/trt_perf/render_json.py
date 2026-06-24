# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serialize extracted TensorRT performance data as JSON."""

from __future__ import annotations

import json
from typing import Any, Dict


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def render_results_json(data: Dict[str, Any], indent: int = 2) -> str:
    return json.dumps(json_ready(data), indent=indent, sort_keys=True) + "\n"
