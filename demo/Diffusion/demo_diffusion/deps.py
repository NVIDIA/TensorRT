#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Dependency management for TensorRT diffusion demos.

Adds the right group's site-packages to sys.path so imports work.
"""

import os
import sys

# Marker file to indicate successful installation
# Must match the marker file used by setup.py
INSTALL_COMPLETE_MARKER = ".install_complete"

# Descriptions for user-facing messages
GROUP_DESCRIPTIONS = {
    "sd": "SD family (SD 1.4, SDXL, SD3, SD3.5, SVD, Stable Cascade)",
    "flux": "Flux family (Black Forest Labs)",
    "cosmos": "Cosmos family (NVIDIA), Wan2.2 T2V",
}

# Valid dependency groups
VALID_GROUPS = list(GROUP_DESCRIPTIONS.keys())

__all__ = [
    "configure",
    "get_configured_groups",
    "print_status",
]


def _resolve_deps_root(deps_root: str | None) -> str:
    """Resolve the dependency root path with precedence: arg > env > default."""
    if deps_root is not None:
        return deps_root
    return os.environ.get("TENSORRT_DIFFUSION_DEPS_ROOT", "/workspace/deps")

def _clean_diffusion_paths(deps_root: str = "/workspace/deps"):
    """Drop any existing paths under deps_root from sys.path."""
    # Filter out any paths that live under deps_root (robust to path forms)
    root_abs = os.path.abspath(os.path.expanduser(deps_root)).rstrip(os.sep) + os.sep
    sys.path[:] = [
        p for p in sys.path
        if not os.path.abspath(os.path.expanduser(p)).startswith(root_abs)
    ]


def configure(
    group: str,
    deps_root: str | None = None,
    verbose: bool = False,
    clean: bool = True,
    fallback: bool = False
):
    """
    Configure sys.path to use dependencies from the specified group.

    This function should be called at the top of each demo script, before
    any other imports that depend on external packages.

    Args:
        group: Dependency group name ("sd", "flux", or "cosmos")
        deps_root: Root directory where dependencies are installed.
                   If None, uses TENSORRT_DIFFUSION_DEPS_ROOT environment variable,
                   or defaults to "/workspace/deps"
        verbose: Print configuration info (default: False)
        clean: Remove other dependency group paths first (default: True)
               This prevents sys.path inflation in long-running processes.
        fallback: If True, gracefully handle missing dependencies instead of raising
                 RuntimeError. Useful for test environments using requirements.txt.
                 If False but USE_REQUIREMENTS=1 is set,
                 fallback will be automatically enabled.

    Example:
        from demo_diffusion import deps
        deps.configure("flux")

        # Or with custom location
        deps.configure("flux", deps_root="/custom/path/deps")

        # Or via environment variable
        # export TENSORRT_DIFFUSION_DEPS_ROOT=/custom/path/deps
        # deps.configure("flux")

        # For test environments using requirements.txt (installed via setup.sh)
        # export USE_REQUIREMENTS=1
        # deps.configure("sd")  # Will automatically use fallback mode
    """
    if group not in VALID_GROUPS:
        raise ValueError(
            f"Invalid dependency group: '{group}'\n"
            f"Valid groups: {', '.join(sorted(VALID_GROUPS))}"
        )

    # Auto-enable fallback mode if using requirements.txt installation
    if not fallback and os.environ.get("USE_REQUIREMENTS") == "1":
        fallback = True
        if verbose:
            print("Detected USE_REQUIREMENTS=1, enabling fallback mode")

    # Resolve deps_root consistently across the module
    deps_root = _resolve_deps_root(deps_root)

    # Clean old dependency paths first (prevents inflation)
    if clean:
        _clean_diffusion_paths(deps_root)

    # Determine Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # Construct path to site-packages for this group
    deps_path = os.path.join(
        deps_root,
        group,
        "lib",
        f"python{python_version}",
        "site-packages"
    )

    # Check if installation is complete
    group_dir = os.path.join(deps_root, group)
    marker_file = os.path.join(group_dir, INSTALL_COMPLETE_MARKER)

    if os.path.exists(deps_path) and os.path.exists(marker_file):
        # Insert at the beginning to override any system packages
        sys.path.insert(0, deps_path)

        if verbose:
            description = GROUP_DESCRIPTIONS.get(group, group)
            print(f"Configured dependencies: {description}")
            print(f"  Path: {deps_path}")
    else:
        # Dependencies not found or installation incomplete
        description = GROUP_DESCRIPTIONS.get(group, group)

        # Check if it's an incomplete installation
        if os.path.exists(group_dir) and not os.path.exists(marker_file):
            error_msg = (
                f"Dependencies for '{group}' are incomplete!\n"
                f"    Location: {group_dir}\n"
                f"    Description: {description}\n"
                f"    A previous installation failed or was interrupted.\n"
                f"    To fix, run:\n"
                f"      python setup.py {group}\n"
                f"    (This will clean up and reinstall)"
            )
            if fallback:
                if verbose:
                    print(f"Warning: {error_msg}")
                    print("Continuing with fallback mode...")
                return
            else:
                raise RuntimeError(error_msg)
        else:
            # Not installed at all
            error_msg = (
                f"Dependencies for '{group}' not found!\n"
                f"    Expected at: {deps_path}\n"
                f"    Description: {description}\n"
                f"    To install, run: python setup.py {group}\n"
                f"    If you installed elsewhere, set TENSORRT_DIFFUSION_DEPS_ROOT or pass deps_root."
            )
            if fallback:
                if verbose:
                    print(f"Warning: {error_msg}")
                    print("Continuing with fallback mode (assuming requirements.txt setup)...")
                return
            else:
                raise RuntimeError(error_msg)


def get_configured_groups(deps_root: str | None = None) -> list[str]:
    """
    Get list of dependency groups that are currently installed.

    A group is considered installed only if both:
    1. The group directory exists
    2. The .install_complete marker file exists in that directory

    Args:
        deps_root: Root directory where dependencies are installed.
                   If None, uses TENSORRT_DIFFUSION_DEPS_ROOT environment variable,
                   or defaults to "/workspace/deps"

    Returns:
        List of installed group names (e.g., ["sd", "flux"])
    """
    # Resolve deps_root
    deps_root = _resolve_deps_root(deps_root)

    installed = []
    for group in VALID_GROUPS:
        # Check if the group directory exists and has the completion marker
        group_dir = os.path.join(deps_root, group)
        marker_file = os.path.join(group_dir, INSTALL_COMPLETE_MARKER)

        # Only consider it installed if marker file exists
        if os.path.exists(group_dir) and os.path.isdir(group_dir) and os.path.exists(marker_file):
            installed.append(group)
    return sorted(installed)


def print_status(deps_root: str | None = None):
    """
    Print the status of all dependency groups.

    Args:
        deps_root: Root directory where dependencies are installed.
                   If None, uses TENSORRT_DIFFUSION_DEPS_ROOT environment variable,
                   or defaults to "/workspace/deps"
    """
    # Resolve deps_root
    deps_root = _resolve_deps_root(deps_root)

    print("Dependency Groups Status:")
    print("-" * 60)
    print(f"Location: {deps_root}")
    print("-" * 60)

    installed = get_configured_groups(deps_root)

    for group in sorted(VALID_GROUPS):
        description = GROUP_DESCRIPTIONS.get(group, group)
        status = "Installed" if group in installed else "Not installed"
        print(f"  {group:8} - {status:15} - {description}")

    print("-" * 60)

    if not installed:
        print("No dependency groups installed.")
        print("Run: python setup.py all")
    else:
        print(f"{len(installed)} group(s) installed: {', '.join(installed)}")
