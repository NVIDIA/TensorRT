#!/usr/bin/env python3
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

"""
Setup script for TensorRT Diffusion dependencies.

Usage:
    python setup.py [GROUP] [OPTIONS]

Groups:
    sd     = SD 1.4, SDXL, SD3, SD3.5, SVD, Stable Cascade (Stability AI)
    flux   = Flux models (Black Forest Labs)
    cosmos = Cosmos models (NVIDIA), Wan2.2 T2V
    all    = Install all groups (default)

Options:
    --skip-tensorrt  Skip TensorRT upgrade/installation
    --force          Force reinstallation even if already installed
    --deps-root DIR  Root directory for dependencies
    -q, --quiet      Suppress informational output (errors are always shown)

Examples:
    python setup.py                       # Install all
    python setup.py sd                    # Install SD family only
    python setup.py flux                  # Install Flux only
    python setup.py cosmos                # Install Cosmos only
    python setup.py --skip-tensorrt       # Install all, skip TensorRT upgrade
    python setup.py flux --skip-tensorrt  # Install Flux only, skip TensorRT upgrade
    python setup.py all --quiet           # Install all, minimal output
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
# Global quiet flag (set by parse_args)
_quiet = False


def log(msg: str = ""):
    """Print a message unless in quiet mode."""
    if not _quiet:
        print(msg)

# Group descriptions
GROUP_DESCRIPTIONS = {
    "sd": "SD family (SD 1.4, SDXL, SD3, SD3.5, SVD, Stable Cascade)",
    "flux": "Flux family (Black Forest Labs)",
    "cosmos": "Cosmos family (NVIDIA), Wan2.2 T2V",
}

# Valid dependency groups
VALID_GROUPS = set(GROUP_DESCRIPTIONS.keys())

# Default installation root
# Can be overridden by:
# 1. Environment variable: TENSORRT_DIFFUSION_DEPS_ROOT
# 2. Command-line argument: --deps-root
DEFAULT_DEPS_ROOT = os.environ.get("TENSORRT_DIFFUSION_DEPS_ROOT", "/workspace/deps")

# Marker file to indicate successful installation
# This file is created after all packages are successfully installed
# Prevents treating incomplete installations as complete
INSTALL_COMPLETE_MARKER = ".install_complete"


def _normalize_package_name(name: str) -> str:
    """Normalize a package name per PEP 503 (e.g. 'Nvidia_ModelOpt' -> 'nvidia-modelopt')."""
    return re.sub(r"[-_.]+", "-", name).lower()


# Regex to extract the package name from a PEP 508 requirement string.
# Matches the leading identifier before any extras, version specifier, or URL marker.
#   "nvidia-modelopt[torch,onnx]==0.40.0"  ->  "nvidia-modelopt"
#   "flux @ git+https://..."               ->  "flux"
_REQ_NAME_RE = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")


def get_pyproject_package_names(pyproject_path: str, group: str) -> set[str]:
    """Return normalized names of packages explicitly listed in pyproject.toml."""
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            return set()

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    reqs = list(project.get("dependencies", []))
    reqs.extend(project.get("optional-dependencies", {}).get(group, []))

    names = set()
    for req in reqs:
        match = _REQ_NAME_RE.match(req.strip())
        if match:
            names.add(_normalize_package_name(match.group(1)))
    return names


def get_container_provided_packages() -> list[str]:
    """Discover torch/NVIDIA packages already installed in the container."""
    from importlib.metadata import distributions
    prefixes = ("torch", "torchvision", "triton", "nvidia-")
    return sorted({
        dist.metadata["Name"].lower()
        for dist in distributions()
        if dist.metadata["Name"].lower().startswith(prefixes)
    })


def print_header():
    """Print setup header."""
    log("=" * 60)
    log("  TensorRT Diffusion - Dependency Setup")
    log("=" * 60)


def check_uv_installed() -> tuple[bool, bool]:
    """
    Check if uv is installed and accessible.

    Returns:
        tuple: (is_installed, needs_path_setup)
            - is_installed: True if uv is available
            - needs_path_setup: True if user needs to add ~/.local/bin to PATH permanently
    """
    # First check if uv is in PATH
    try:
        subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            check=True,
            text=True
        )
        return True, False
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If not in PATH, check the default installation location
        local_bin = os.path.expanduser("~/.local/bin")
        uv_path = os.path.join(local_bin, "uv")

        if os.path.exists(uv_path) and os.access(uv_path, os.X_OK):
            # uv exists but not in PATH - add it temporarily for this script
            if local_bin not in os.environ["PATH"]:
                os.environ["PATH"] = f"{local_bin}:{os.environ['PATH']}"
            return True, True  # Needs permanent PATH setup

        return False, False


def upgrade_tensorrt(pip_spec: str) -> bool:
    """Upgrade/install TensorRT using pip.

    Args:
        pip_spec: PIP specifier for TensorRT, e.g. "tensorrt-cu12" or "tensorrt-cu12==10.16.*"

    Returns:
        True on success, False otherwise
    """
    print("Installing TensorRT: {}".format(pip_spec))
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "--pre", pip_spec],
            check=True,
            capture_output=_quiet,
        )
        # Print installed version for confirmation
        try:
            import importlib
            trt = importlib.import_module("tensorrt")
            log(f"   Installed TensorRT version: {getattr(trt, '__version__', 'unknown')}")
        except Exception:
            pass
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to upgrade/install TensorRT: {e}")
        return False


def install_libgl1() -> bool:
    """Install libgl1 system package when possible (Debian/Ubuntu).

    Returns:
        True if successfully installed or already present, False otherwise.
    """
    print("Checking for libgl1 (system dependency)...")
    apt_get = shutil.which("apt-get")
    if not apt_get:
        log("   Skipping: apt-get not found. Please install 'libgl1' via your OS package manager.")
        return False

    sudo = shutil.which("sudo")
    use_sudo = False
    try:
        use_sudo = (hasattr(os, "geteuid") and os.geteuid() != 0 and sudo is not None)
    except Exception:
        use_sudo = sudo is not None

    try:
        # Update package list first
        cmd_update = ([sudo] if use_sudo else []) + [apt_get, "update"]
        subprocess.run(cmd_update, check=True, capture_output=_quiet)

        cmd_install = ([sudo] if use_sudo else []) + [apt_get, "install", "-y", "libgl1"]
        subprocess.run(cmd_install, check=True, capture_output=_quiet)
        log("   libgl1 installed (or already up to date)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   Warning: Failed to install libgl1: {e}")
        print("   Please install 'libgl1' manually via your OS package manager.")
        return False


def install_uv():
    """Install uv package manager."""
    log("Installing uv...")
    try:
        # Download and run uv installer
        curl_cmd = [
            "curl", "-LsSf",
            "https://astral.sh/uv/install.sh"
        ]

        curl_process = subprocess.Popen(
            curl_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        sh_process = subprocess.Popen(
            ["sh"],
            stdin=curl_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        curl_process.stdout.close()
        stdout, stderr = sh_process.communicate()

        if sh_process.returncode != 0:
            print(f"Failed to install uv: {stderr.decode()}")
            sys.exit(1)

        # Add to PATH for this session
        # uv installer uses ~/.local/bin by default
        local_bin = os.path.expanduser("~/.local/bin")
        if local_bin not in os.environ["PATH"]:
            os.environ["PATH"] = f"{local_bin}:{os.environ['PATH']}"
        log("uv installed successfully")

    except Exception as e:
        print(f"Error installing uv: {e}")
        print("Please install manually: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)


def get_installed_groups(deps_root: str) -> set[str]:
    """
    Get set of already installed dependency groups.

    A group is considered installed only if both:
    1. The group directory exists
    2. The .install_complete marker file exists in that directory

    Args:
        deps_root: Root directory where dependencies are installed

    Returns:
        Set of installed group names
    """
    installed = set()
    deps_path = Path(deps_root)

    if not deps_path.exists():
        return installed

    for group in VALID_GROUPS:
        group_dir = deps_path / group
        marker_file = group_dir / INSTALL_COMPLETE_MARKER

        # Only consider it installed if marker file exists
        if group_dir.exists() and marker_file.exists():
            installed.add(group)

    return installed


def install_group(group: str, deps_root: str, project_root: str) -> bool:
    """
    Install dependencies for a specific group.

    Args:
        group: Dependency group name
        deps_root: Root directory where dependencies will be installed
        project_root: Project root directory (where pyproject.toml is)

    Returns:
        True if installation succeeded, False otherwise
    """
    description = GROUP_DESCRIPTIONS.get(group, group)
    install_path = os.path.join(deps_root, group)

    print(f"Installing {description}...")
    log(f"   Location: {install_path}")

    try:
        # Determine Python version and site-packages path
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        site_packages = Path(install_path) / "lib" / f"python{python_version}" / "site-packages"

        # Create site-packages directory if it doesn't exist
        site_packages.mkdir(parents=True, exist_ok=True)

        # Build an overrides file so uv never installs packages that are
        # already installed in the container. Exclude packages explicitly
        # listed in pyproject.toml so they get installed at the pinned version.
        container_pkgs = get_container_provided_packages()
        declared_pkgs = get_pyproject_package_names(
            str(Path(project_root) / "pyproject.toml"), group
        )
        overrides_content = "\n".join(
            f'{pkg} ; python_version < "0"'
            for pkg in container_pkgs
            if _normalize_package_name(pkg) not in declared_pkgs
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="uv_overrides_", delete=False
        ) as f:
            f.write(overrides_content)
            overrides_path = f.name

        try:
            cmd = [
                "uv", "pip", "install",
                "--python-preference", "only-system",
                "--prefix", install_path,
                "--overrides", overrides_path,
                f".[{group}]"
            ]

            subprocess.run(
                cmd,
                cwd=project_root,
                text=True,
                capture_output=_quiet,
                check=True
            )
        finally:
            os.unlink(overrides_path)

        # Create marker file to indicate successful installation
        marker_file = Path(install_path) / INSTALL_COMPLETE_MARKER
        marker_file.write_text("Installation completed successfully\n")

        print(f"   {description} installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed to install {description}")
        print(f"   Error: {e}")

        # Clean up incomplete installation
        if os.path.exists(install_path):
            log(f"   Cleaning up incomplete installation at {install_path}")
            try:
                shutil.rmtree(install_path)
            except Exception as cleanup_error:
                print(f"   Warning: Failed to clean up: {cleanup_error}")

        return False


def determine_groups_to_install(requested: str, already_installed: set[str]) -> list[str]:
    """
    Determine which groups need to be installed.

    Args:
        requested: Requested group or "all"
        already_installed: Set of already installed groups

    Returns:
        List of groups to install
    """
    if requested == "all":
        return sorted(VALID_GROUPS - already_installed)
    elif requested in VALID_GROUPS:
        return [] if requested in already_installed else [requested]
    return []


def print_summary(installed_groups: set[str]):
    """
    Print summary of installed groups.

    Args:
        installed_groups: Set of all installed groups
    """
    log("=" * 60)
    print("Setup complete!")
    log("=" * 60)

    if installed_groups:
        log("Installed groups:")
        for group in sorted(installed_groups):
            description = GROUP_DESCRIPTIONS.get(group, group)
            log(f"   {description}")

        log("Each demo script automatically uses the correct dependencies.")
    else:
        log("No groups installed.")

    log()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup TensorRT Diffusion dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py              # Install all groups
  python setup.py sd           # Install SD family only
  python setup.py flux         # Install Flux only
  python setup.py cosmos       # Install Cosmos only

Groups:
  sd     - SD 1.4, SDXL, SD3, SD3.5, SVD, Stable Cascade (Stability AI)
  flux   - Flux models (Black Forest Labs)
  cosmos - Cosmos models (NVIDIA)
  all    - All groups (default)
        """
    )

    parser.add_argument(
        "group",
        nargs="?",
        default="all",
        choices=list(VALID_GROUPS) + ["all"],
        help="Dependency group to install (default: all)"
    )

    parser.add_argument(
        "--deps-root",
        default=DEFAULT_DEPS_ROOT,
        help=f"Root directory for dependencies (default: {DEFAULT_DEPS_ROOT})"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstallation even if already installed"
    )

    parser.add_argument("--skip-tensorrt", action="store_true", help="Skip TensorRT upgrade/installation")

    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress informational output (errors are always shown)")

    return parser.parse_args()


def main():
    """Main setup function."""
    args = parse_args()

    global _quiet
    _quiet = args.quiet

    print_header()

    # Check/install uv
    uv_installed, needs_path_setup = check_uv_installed()
    if not uv_installed:
        install_uv()
        needs_path_setup = True  # Fresh install always needs PATH setup
    else:
        log("uv is already installed")
        if needs_path_setup:
            log("   (temporarily added ~/.local/bin to PATH for this session)")

    # Get project root (where pyproject.toml is)
    project_root = str(Path(__file__).parent.absolute())
    pyproject_path = Path(project_root) / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"Error: pyproject.toml not found in {project_root}")
        print("   Make sure you're running this script from the project root.")
        sys.exit(1)

    # Check what's already installed
    installed_groups = get_installed_groups(args.deps_root)

    # Install system dependency (best-effort) and ensure TensorRT is present
    install_libgl1()
    if not args.skip_tensorrt:
        upgrade_tensorrt("tensorrt-cu12")
    else:
        log("Skipping TensorRT upgrade (--skip-tensorrt specified)")

    if installed_groups and not args.force:
        log(f"Already installed: {', '.join(sorted(installed_groups))}")

    # Determine what to install
    if args.force and args.group == "all":
        groups_to_install = sorted(VALID_GROUPS)
    elif args.force and args.group in VALID_GROUPS:
        groups_to_install = [args.group]
    else:
        groups_to_install = determine_groups_to_install(args.group, installed_groups)

    # Early exit if nothing to do
    if not groups_to_install:
        if args.group == "all":
            print("All requested dependencies are already installed!")
        else:
            description = GROUP_DESCRIPTIONS.get(args.group, args.group)
            print(f"{description} is already installed!")
        log(f"   To reinstall, use --force flag or remove {args.deps_root}/{args.group}")
        print_summary(installed_groups)
        sys.exit(0)

    # Install groups
    print(f"Installing {len(groups_to_install)} group(s): {', '.join(groups_to_install)}")

    success_count = 0
    for group in groups_to_install:
        if install_group(group, args.deps_root, project_root):
            installed_groups.add(group)
            success_count += 1

    # Print summary
    print_summary(installed_groups)

    # Print PATH setup instructions if needed
    if needs_path_setup:
        print()
        print("=" * 60)
        print("  Action Required: Update PATH")
        print("=" * 60)
        print("uv was installed to ~/.local/bin but is not in your PATH.")
        print()
        print("To make uv available in future sessions, run:")
        print("  source ~/.local/bin/env")
        print()
        print("Or restart your shell.")
        print("=" * 60)

    # Exit with error if any installations failed
    if success_count < len(groups_to_install):
        print(f"Warning: {len(groups_to_install) - success_count} group(s) failed to install")
        sys.exit(1)


if __name__ == "__main__":
    main()
