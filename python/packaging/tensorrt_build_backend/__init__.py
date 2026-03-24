#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
TensorRT Custom Build Backend

A PEP 517 compliant build backend that wraps setuptools to provide:
1. Custom wheel tag support (python-tag, plat-name) via CLI config settings
2. Dynamic package naming based on wheel type (standalone, libs, frontend, etc.)

This backend reads configuration from [tool.tensorrt] in pyproject.toml and
computes the appropriate package names based on the wheel-type config setting
passed via the build frontend CLI.

Usage in pyproject.toml:
    [build-system]
    requires = ["setuptools>=61.0", "wheel"]
    build-backend = "tensorrt_build_backend"
    backend-path = ["."]

    [tool.tensorrt]
    base-name = "tensorrt"        # Base module name
    cuda-major = "12"             # CUDA major version

Config settings (passed via `python -m build --config-setting=key=value`):
    wheel-type: Type of wheel being built. One of:
        - "binding": Non-standalone bindings (name = base-name)
        - "binding_standalone": Standalone bindings (name = {base-name}_cu{cuda}_bindings)
        - "libs": Libraries wheel (name = {base-name}_cu{cuda}_libs)
        - "frontend": Frontend/metapackage (name = {base-name}_cu{cuda})
    python-tag: Python tag for the wheel (e.g., "cp312")
    plat-name: Platform tag for the wheel (e.g., "linux_x86_64")
"""

import shutil
import tempfile
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from setuptools.build_meta import (
    build_sdist,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_wheel,
)
from setuptools.build_meta import build_wheel as _setuptools_build_wheel


def _read_pyproject_toml():
    """Read and parse pyproject.toml from the current directory."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return {}
    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)


def _get_tensorrt_config():
    """
    Get TensorRT-specific configuration from pyproject.toml.
    
    Returns a dict with keys:
        - base_name: Base module name (e.g., "tensorrt")
        - cuda_major: CUDA major version (e.g., "12")
        - version: Package version (e.g., "10.0.0.1")
    """
    pyproject = _read_pyproject_toml()
    trt_config = pyproject.get("tool", {}).get("tensorrt", {})
    
    return {
        "base_name": trt_config.get("base-name", "tensorrt"),
        "cuda_major": trt_config.get("cuda-major", "12"),
        "version": trt_config.get("version") or pyproject.get("project", {}).get("version", "0.0.0"),
    }


def _compute_package_names(trt_config, wheel_type):
    """
    Compute the distribution and import package names based on wheel type.
    
    Args:
        trt_config: Dict with base_name, cuda_major, version
        wheel_type: One of "binding", "binding_standalone", "libs", "frontend"
        
    Returns:
        tuple: (distribution_name, import_name)
            - distribution_name: Name for pip install (e.g., "tensorrt_cu12_bindings")
            - import_name: Name for Python import (e.g., "tensorrt_bindings")
    """
    base_name = trt_config["base_name"]
    cuda_major = trt_config["cuda_major"]
    
    if wheel_type == "binding_standalone":
        # Standalone bindings: tensorrt_cu12_bindings / tensorrt_bindings
        return f"{base_name}_cu{cuda_major}_bindings", f"{base_name}_bindings"
    elif wheel_type == "libs":
        # Libs wheel: tensorrt_cu12_libs / tensorrt
        return f"{base_name}_cu{cuda_major}_libs", base_name
    elif wheel_type == "frontend":
        # Frontend wheel: tensorrt_cu12 / tensorrt
        return f"{base_name}_cu{cuda_major}", base_name
    else:
        # Non-standalone binding or default: tensorrt / tensorrt
        return base_name, base_name


def _get_wheel_tags(config_settings):
    """
    Extract wheel tags from config_settings.
    
    Returns:
        tuple: (python_tag, plat_name) or (None, None) if not specified
    """
    config_settings = config_settings or {}
    
    python_tag = config_settings.get("python-tag")
    plat_name = config_settings.get("plat-name")
    
    return python_tag, plat_name


def _get_wheel_type(config_settings):
    """
    Get the wheel type from config_settings.
    
    Returns one of: "binding", "binding_standalone", "libs", "frontend", or None
    Returns None if not set (passthrough mode - no dynamic naming).
    """
    config_settings = config_settings or {}
    return config_settings.get("wheel-type")


def _update_pyproject_for_build(dist_name, import_name):
    """
    Update pyproject.toml in-place with computed package name and version.
    
    This allows setuptools to pick up the correct dynamic values.
    We modify the file temporarily during the build process.
    
    Args:
        dist_name: Distribution name (for pip install)
        import_name: Import name (for Python import)  
    """
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Update name field - replace placeholder or any existing name
    content = content.replace(
        'name = "build-backend-placeholder"',
        f'name = "{dist_name}"'
    )
    
    # Update packages.find.include to use import name
    content = content.replace(
        'include = ["build-backend-placeholder"]',
        f'include = ["{import_name}*"]'
    )
    
    pyproject_path.write_text(content)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """
    Build a wheel with TensorRT-specific customizations.
    
    This is the main PEP 517 hook for building wheels. It:
    1. If wheel-type config setting is set: computes package name dynamically and updates pyproject.toml
    2. Applies custom python/platform tags if specified via config settings
    3. Builds the wheel using setuptools
    4. Moves the final wheel to the output directory
    
    Config settings (passed via --config-setting):
        wheel-type: Type of wheel being built (binding, binding_standalone, libs, frontend)
        python-tag: Python tag for the wheel (e.g., cp312)
        plat-name: Platform tag for the wheel (e.g., linux_x86_64)
    
    When wheel-type is not set, this acts as a passthrough to setuptools,
    only applying wheel tags if specified.
    """
    wheel_type = _get_wheel_type(config_settings)
    
    # Only do dynamic naming if wheel type is specified
    if wheel_type:
        trt_config = _get_tensorrt_config()
        dist_name, import_name = _compute_package_names(trt_config, wheel_type)
        _update_pyproject_for_build(dist_name, import_name)
    
    # Get wheel tags
    python_tag, plat_name = _get_wheel_tags(config_settings)
    
    # Build config_settings to pass wheel tags directly to setuptools
    build_config = dict(config_settings or {})
    
    # Pass wheel options via --build-option (setuptools.build_meta convention)
    build_options = []
    if python_tag:
        build_options.append(f"--python-tag={python_tag}")
    if plat_name:
        build_options.append(f"--plat-name={plat_name}")
    
    if build_options:
        # Merge with any existing build options
        existing = build_config.get("--build-option", [])
        if isinstance(existing, str):
            existing = [existing]
        build_config["--build-option"] = list(existing) + build_options
    
    # Build wheel using setuptools to a temporary directory first
    # This avoids conflicts with existing wheels in the output directory
    with tempfile.TemporaryDirectory() as tmp_wheel_dir:
        wheel_filename = _setuptools_build_wheel(
            tmp_wheel_dir,
            config_settings=build_config,
            metadata_directory=metadata_directory
        )
        
        wheel_path = Path(tmp_wheel_dir) / wheel_filename
        
        # Move the final wheel to the output directory
        final_wheel_path = Path(wheel_directory) / wheel_path.name
        
        # Remove existing wheel if present (handles parallel build conflicts)
        if final_wheel_path.exists():
            final_wheel_path.unlink()
        
        shutil.move(str(wheel_path), str(final_wheel_path))
        
        return final_wheel_path.name


# Re-export other hooks from setuptools.build_meta
__all__ = [
    "build_wheel",
    "build_sdist", 
    "get_requires_for_build_wheel",
    "get_requires_for_build_sdist",
    "prepare_metadata_for_build_wheel",
]
