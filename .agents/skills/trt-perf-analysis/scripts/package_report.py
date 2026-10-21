#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a standalone TensorRT performance report folder.

This script keeps report packaging deterministic and cross-platform: it runs
the analyzer, validates the generated JSON, then copies the browser report
template without relying on shell tools such as rsync.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from trt_perf.data import infer_model_name_metadata
from trt_perf.license_text import APACHE_2_0


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def skill_root() -> Path:
    return script_dir().parent


def default_template_dir() -> Path:
    return skill_root() / "report_template"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and package a standalone TensorRT performance report."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Folder directly containing matching layers_*.json and profile_*.json inputs. Pass one component/report folder per invocation. Omit when using --analyze-data.",
    )
    parser.add_argument(
        "--analyze-data",
        help="Existing analyzer JSON to validate and package. Requires --report-dir.",
    )
    parser.add_argument(
        "--output-parent",
        help="Directory where an auto-named report folder should be created for folder input. Cannot be combined with --report-dir.",
    )
    parser.add_argument(
        "--report-dir",
        help="Exact report directory to create. It must be empty or absent; required with --analyze-data.",
    )
    parser.add_argument(
        "--model-name",
        help="Model name for folder analysis. It is cleaned and stored in analyze-data.json.",
    )
    parser.add_argument(
        "--timestamp",
        help="Timestamp for auto-generated folder-analysis reports, formatted as YYYYMMDD_HHMMSS. Defaults to local time.",
    )
    parser.add_argument(
        "--template-dir",
        default=str(default_template_dir()),
        help="Report template directory. Defaults to report_template in this skill.",
    )
    parser.add_argument(
        "--analyze-md",
        help="AI-generated Markdown analysis to copy into the report folder as analyze.md after the final analysis is written.",
    )
    return parser.parse_args(argv)


def input_source_paths(input_path: Path) -> list[str]:
    paths: list[Path] = []
    for pattern in ("layers_*.json", "profile_*.json"):
        paths.extend(input_path.glob(pattern))
    return [str(path) for path in sorted(paths)]


def inferred_analyzer_model_name(input_path: Path, explicit_name: Optional[str]) -> Optional[str]:
    metadata = infer_model_name_metadata(str(input_path), explicit_name, input_source_paths(input_path))
    return metadata["name"]


def infer_report_model_name(input_path: Path, explicit_name: Optional[str]) -> str:
    metadata = infer_model_name_metadata(str(input_path), explicit_name, input_source_paths(input_path))
    return metadata["name"] or "unknown_model"


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def default_output_parent() -> Path:
    cwd = Path.cwd().resolve()
    root = skill_root().resolve()
    if is_relative_to(cwd, root):
        return Path(tempfile.gettempdir()).resolve()
    return cwd


def parse_timestamp(value: Optional[str]) -> str:
    if value is None:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{8}_\d{6}", value):
        raise ValueError("--timestamp must use YYYYMMDD_HHMMSS format")
    return value


def has_files(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


def has_input_files(path: Path) -> bool:
    return any(path.glob("layers_*.json")) or any(path.glob("profile_*.json"))


def next_available_report_dir(output_parent: Path, model_name: str, timestamp: str) -> Path:
    base_name = f"report_{model_name}_{timestamp}"
    candidate = output_parent / base_name
    if not has_files(candidate):
        return candidate

    index = 2
    while True:
        candidate = output_parent / f"{base_name}_{index}"
        if not has_files(candidate):
            return candidate
        index += 1


def resolve_new_report_dir(args: argparse.Namespace, input_path: Path) -> Path:
    if args.report_dir:
        report_dir = Path(args.report_dir).expanduser()
        if has_files(report_dir):
            raise ValueError(f"report directory is not empty: {report_dir}")
        return report_dir

    output_parent = Path(args.output_parent).expanduser() if args.output_parent else default_output_parent()
    timestamp = parse_timestamp(args.timestamp)
    model_name = infer_report_model_name(input_path, args.model_name)
    return next_available_report_dir(output_parent, model_name, timestamp)


def validate_template_dir(template_dir: Path) -> None:
    required = [
        "index.html",
        "app.js",
        "netron-assets.js",
        "styles.css",
        # Every report redistributes the bundled third-party code, so it must
        # carry those terms. The Apache-2.0 text is written from
        # trt_perf.license_text rather than copied from the template.
        "ThirdPartyNotices.txt",
    ]
    missing = [name for name in required if not (template_dir / name).is_file()]
    if missing:
        raise ValueError(f"report template is missing required file(s): {', '.join(missing)}")


def run_command(command: Sequence[str]) -> int:
    process = subprocess.run(command, text=True, check=False)
    return process.returncode


def run_analyzer(input_path: Path, output_path: Path, model_name: Optional[str]) -> int:
    analyzer = script_dir() / "analyze_trt_perf.py"
    command = [
        sys.executable,
        str(analyzer),
        str(input_path),
        "--output",
        str(output_path),
    ]
    if model_name:
        command.extend(["--model-name", model_name])
    return run_command(command)


def run_validator(analyze_data_path: Path, require_passed: bool = False) -> int:
    validator = script_dir() / "validate_analyze_data.py"
    command = [sys.executable, str(validator), str(analyze_data_path)]
    if require_passed:
        command.append("--require-passed")
    return run_command(command)


def stage_report_assets(template_dir: Path, report_dir: Path) -> None:
    validate_template_dir(template_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    for item in template_dir.iterdir():
        if item.name in {"analyze-data.json", "analyze.md"}:
            continue

        destination = report_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)

    # Written here rather than copied from the template so the published skill
    # tree has no bare LICENSE file for a scanner to misread as a directory-wide
    # declaration. Every staged report still ships one.
    (report_dir / "LICENSE").write_text(APACHE_2_0, encoding="utf-8")


def copy_analyze_markdown(analyze_md: Optional[str], report_dir: Path) -> Optional[Path]:
    if not analyze_md:
        return None

    source = Path(analyze_md).expanduser()
    if not source.is_file():
        raise ValueError(f"analyze markdown file not found: {source}")

    destination = report_dir / "analyze.md"
    if source.resolve() == destination.resolve():
        return destination

    shutil.copy2(source, destination)
    return destination


def publish_staging_dir(staging_dir: Path, report_dir: Path) -> None:
    if report_dir.exists():
        if has_files(report_dir):
            raise ValueError(f"report directory became non-empty: {report_dir}")
        report_dir.rmdir()
    staging_dir.replace(report_dir)


def package_existing_json(args: argparse.Namespace) -> int:
    analyze_data_path = Path(args.analyze_data).expanduser()
    if not analyze_data_path.is_file():
        sys.stderr.write(f"error: analyze-data file not found: {analyze_data_path}\n")
        return 2

    if not args.report_dir:
        sys.stderr.write("error: --report-dir is required with --analyze-data\n")
        return 2
    report_dir = Path(args.report_dir).expanduser()
    if has_files(report_dir):
        sys.stderr.write(f"error: report directory is not empty: {report_dir}\n")
        return 2

    validate_result = run_validator(analyze_data_path, require_passed=True)
    if validate_result != 0:
        return validate_result

    try:
        report_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".trt-perf-analysis-", dir=report_dir.parent) as temp_dir:
            staging_dir = Path(temp_dir) / "report"
            staging_dir.mkdir()
            shutil.copy2(analyze_data_path, staging_dir / "analyze-data.json")
            stage_report_assets(Path(args.template_dir).expanduser(), staging_dir)
            staged_markdown_path = copy_analyze_markdown(args.analyze_md, staging_dir)
            publish_staging_dir(staging_dir, report_dir)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"error: unable to package report template: {exc}\n")
        return 2

    packaged_data_path = report_dir / "analyze-data.json"
    print(f"report_dir: {report_dir}")
    print(f"analyze_data: {packaged_data_path}")
    if staged_markdown_path:
        print(f"analyze_md: {report_dir / 'analyze.md'}")
    return 0


def package_input_folder(args: argparse.Namespace, input_path: Path) -> int:
    try:
        report_dir = resolve_new_report_dir(args, input_path)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    try:
        report_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sys.stderr.write(f"error: unable to create report parent directory: {exc}\n")
        return 2

    with tempfile.TemporaryDirectory(prefix=".trt-perf-analysis-", dir=report_dir.parent) as temp_dir:
        staging_dir = Path(temp_dir) / "report"
        staging_dir.mkdir()
        staging_data_path = staging_dir / "analyze-data.json"

        model_name = inferred_analyzer_model_name(input_path, args.model_name)
        analyze_result = run_analyzer(input_path, staging_data_path, model_name)
        if analyze_result != 0:
            return analyze_result

        validate_result = run_validator(staging_data_path, require_passed=True)
        if validate_result != 0:
            return validate_result

        try:
            stage_report_assets(Path(args.template_dir).expanduser(), staging_dir)
            staged_markdown_path = copy_analyze_markdown(args.analyze_md, staging_dir)
            publish_staging_dir(staging_dir, report_dir)
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"error: unable to package report template: {exc}\n")
            return 2

    analyze_data_path = report_dir / "analyze-data.json"
    analyze_markdown_path = report_dir / "analyze.md" if staged_markdown_path else None

    print(f"report_dir: {report_dir}")
    print(f"analyze_data: {analyze_data_path}")
    if analyze_markdown_path:
        print(f"analyze_md: {analyze_markdown_path}")
    return 0


def package_new_analysis(args: argparse.Namespace) -> int:
    if not args.path:
        sys.stderr.write("error: provide an input folder, or use --analyze-data\n")
        return 2

    input_path = Path(args.path).expanduser()
    if not input_path.is_dir():
        sys.stderr.write(f"error: input folder not found: {input_path}\n")
        return 2

    if has_input_files(input_path):
        return package_input_folder(args, input_path)

    sys.stderr.write(
        f"error: no layers_*.json or profile_*.json files found directly in {input_path}. "
        "package_report.py creates exactly one report per invocation; pass a single "
        "component/report folder instead of a parent folder.\n"
    )
    return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.path and args.analyze_data:
        sys.stderr.write("error: use either an input folder or --analyze-data, not both\n")
        return 2
    if args.report_dir and args.output_parent:
        sys.stderr.write("error: use either --report-dir or --output-parent, not both\n")
        return 2
    if args.analyze_data and (args.output_parent or args.model_name or args.timestamp):
        sys.stderr.write(
            "error: --output-parent, --model-name, and --timestamp apply only to folder analysis\n"
        )
        return 2

    if args.analyze_data:
        return package_existing_json(args)
    return package_new_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
