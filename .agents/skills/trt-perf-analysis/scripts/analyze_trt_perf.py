#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and extract structured TensorRT layer/profile analysis data.

The script intentionally uses only Python built-in modules so it can run in
minimal benchmark environments.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from trt_perf.data import DataError, extract_perf_data
from trt_perf.render_json import render_results_json


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and analyze TensorRT layer/profile JSON performance data."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Folder directly containing one or more layers_*.json/profile_*.json backend pairs. Omit when using --data.",
    )
    parser.add_argument(
        "--data",
        dest="data_specs",
        action="append",
        nargs="+",
        metavar="PATH",
        help="Explicit backend input. Use --data layers.json for layer-only analysis or --data layers.json profile.json for a layer/profile pair. Repeat for multiple backends.",
    )
    parser.add_argument(
        "--model-name",
        help="Optional model name from the user prompt or caller context. It is cleaned before serialization.",
    )
    parser.add_argument("--output", help="Optional path to write the generated analysis JSON.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        data = extract_perf_data(args.path, args.data_specs, args.model_name)
    except DataError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    report = render_results_json(data)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(report)
        except OSError as exc:
            sys.stderr.write(f"error: unable to write {args.output}: {exc}\n")
            return 2
    else:
        sys.stdout.write(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
