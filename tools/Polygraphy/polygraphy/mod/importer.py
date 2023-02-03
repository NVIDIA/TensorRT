#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import importlib
import os
import subprocess as sp
import sys
import warnings
from typing import List

from polygraphy import constants
from polygraphy.mod import util as mod_util

# Tracks all of Polygraphy's lazy imports, excluding internal ones.
_all_external_lazy_imports = set()

# Sometimes the Python package name differs from the module name.
_PKG_NAME_FROM_MODULE = {
}

# Some packages need additional flags to install correctly.
_EXTRA_FLAGS_FOR_MODULE = {
    "onnx_graphsurgeon": ["--extra-index-url=https://pypi.ngc.nvidia.com"],
}


LATEST_VERSION = "==latest"
"""Indicates that the latest version of the package is preferred in lazy_import"""


def _version_ok(ver, preferred):
    if preferred == LATEST_VERSION:
        return False

    pref_ver = preferred.lstrip("<=>").strip()
    cond = preferred.rstrip(pref_ver).strip()
    check = {
        "==": lambda x, y: x == y,
        ">=": lambda x, y: x >= y,
        ">": lambda x, y: x > y,
        "<=": lambda x, y: x <= y,
        "<": lambda x, y: x < y,
    }[cond]
    return check(mod_util.version(ver), mod_util.version(pref_ver))


def lazy_import(
    name: str,
    log: bool = None,
    version: str = None,
    pkg_name: str = None,
    install_flags: List[str] = None,
    requires: List[str] = None,
):
    """
    Lazily import a module.

    If config.AUTOINSTALL_DEPS is set to 1,
    missing modules are automatically installed, and existing modules may be
    upgraded if newer versions are required.

    Args:
        name (str):
                The name of the module and optionally the preferred version of the package,
                formatted as a version string. For example, ``'example_module>=0.5.0'`` or ``'example_module==1.8.0'``.
        log (bool):
                Whether to log information about the module.
                Defaults to True.
        version (str):
                [DEPRECATED - use `name` instead]
                The preferred version of the package, formatted as a version string.
                For example, ``'>=0.5.0'`` or ``'==1.8.0'``. Use ``LATEST_VERSION`` to
                indicate that the latest version of the package is preferred.
        pkg_name (str):
                The name of the package that provides this module, if it is different from the module name.
                Used only if automatic installation of dependencies is enabled.
        install_flags (List[str]):
                Additional flags to provide to the installation command.
                Used only if automatic installation of dependencies is enabled.
        requires (List[str]):
                Additional dependencies required by the module which are *not* specified as dependencies.
                This parameter should only be required when a module does not correctly specify dependencies.
                Defaults to [].

    Returns:
        LazyModule:
                A lazily loaded module. When an attribute is first accessed,
                the module will be imported.
    """
    VERSION_CHARS = ["=", ">", "<"]
    assert (
        version is None or version == LATEST_VERSION or any(version.startswith(char) for char in VERSION_CHARS)
    ), "version must be formatted as a version string!"

    if version is not None:
        warnings.warn(
            "The version parameter in lazy_import is deprecated and will be removed in v0.45.0.\n"
            "The version can instead be specified as part of the package name.",
            DeprecationWarning,
            stacklevel=3,
        )

    log = True if log is None else log
    requires = [] if requires is None else requires

    def split_name_version(inp):
        version_char_indices = [inp.index(char) for char in VERSION_CHARS if char in inp]
        if not version_char_indices:
            return inp, None

        min_index = min(version_char_indices)
        return inp[:min_index], inp[min_index:]

    name, version = split_name_version(name) if version is None else (name, version)
    all_required_mods = list(map(split_name_version, requires)) + [(name, version)]

    if "polygraphy" not in name:
        _all_external_lazy_imports.add(name)

    def import_mod():
        from polygraphy import config
        from polygraphy.logger import G_LOGGER, LogMode

        def install_mod(install_name, install_version, raise_error=True):
            modname = install_name.split(".")[0]
            pkg = pkg_name if pkg_name is not None else _PKG_NAME_FROM_MODULE.get(modname, modname)
            extra_flags = install_flags if install_flags is not None else _EXTRA_FLAGS_FOR_MODULE.get(modname, [])

            def fail():
                log_func = G_LOGGER.critical if raise_error else G_LOGGER.warning
                log_func(f"Could not automatically install required module: {pkg}. Please install it manually.")

            if config.ASK_BEFORE_INSTALL:
                res = None
                while res not in ["y", "n"]:
                    res = input(f"Automatically install '{pkg}' (version: {install_version or 'any'}) ([Y]/n)? ")
                    res = res.strip()[:1].lower() or "y"

                if res == "n":
                    fail()

            if install_version == LATEST_VERSION:
                extra_flags.append("--upgrade")
            elif install_version is not None:
                pkg += install_version

            cmd = config.INSTALL_CMD + [pkg] + extra_flags
            G_LOGGER.info(f"Running installation command: {' '.join(cmd)}")
            status = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            if status.returncode != 0:
                G_LOGGER.error(f"Error during installation:\n{constants.TAB}{status.stderr.decode()}")
                fail()

            mod = importlib.import_module(install_name)
            return mod

        mod = None
        try:
            mod = importlib.import_module(name)
        except ImportError as err:
            if config.AUTOINSTALL_DEPS:
                for install_name, install_version in all_required_mods:
                    G_LOGGER.info(
                        f"Module: '{install_name}' is required, but not installed. Attempting to install now."
                    )
                    mod = install_mod(install_name, install_version)
            else:
                G_LOGGER.critical(
                    f"Module: '{name}' is required but could not be imported.\nNote: Error was: {err}\n"
                    f"You can set POLYGRAPHY_AUTOINSTALL_DEPS=1 in your environment variables to allow "
                    f"Polygraphy to automatically install missing modules.\n"
                )

        # Auto-upgrade if necessary
        for install_name, install_version in all_required_mods:
            installed_mod = importlib.import_module(install_name)
            if (
                install_version is not None
                and hasattr(installed_mod, "__version__")
                and not _version_ok(installed_mod.__version__, install_version)
            ):
                if config.AUTOINSTALL_DEPS:
                    G_LOGGER.info(
                        f"Note: Module: '{install_name}' version '{installed_mod.__version__}' is installed, but version '{install_version}' is required.\n"
                        f"Attempting to upgrade now."
                    )
                    # We can try to use the other version if install fails, so this is non-fatal.
                    installed_mod = install_mod(install_name, install_version, raise_error=False)
                    if install_name == name:
                        mod = installed_mod

                elif install_version != LATEST_VERSION:
                    G_LOGGER.error(
                        f"Module: '{install_name}' version '{installed_mod.__version__}' is installed, but version '{install_version}' is required.\n"
                        f"Please install the required version or set POLYGRAPHY_AUTOINSTALL_DEPS=1 in your environment variables "
                        f"to allow Polygraphy to do so automatically.\n"
                        f"Attempting to continue with the currently installed version of this module, but note that this may cause errors!",
                        mode=LogMode.ONCE,
                    )

        if log:
            G_LOGGER.module_info(mod)

        return mod

    MODULE_VAR_NAME = "module"

    class LazyModule:
        def __init__(self):
            super().__setattr__(MODULE_VAR_NAME, None)

        def __polygraphy_import_mod(self):
            if self.module is None:
                super().__setattr__(MODULE_VAR_NAME, import_mod())
            return self.module

        def __getattr__(self, name):
            module = self.__polygraphy_import_mod()
            return getattr(module, name)

        def __setattr__(self, name, value):
            module = self.__polygraphy_import_mod()
            return setattr(module, name, value)

    return LazyModule()


def has_mod(modname):
    """
    Checks whether a module is available and usable.

    Args:
        modname (str): The name of the module to check.

    Returns:
        bool:
                Whether the module is available and usable.
                This returns false if importing the module fails for any reason.
    """
    try:
        importlib.import_module(modname)
    except:
        return False
    return True


def autoinstall(lazy_mod):
    """
    If the config.AUTOINSTALL_DEPS is set to 1, automatically install or upgrade a module.
    Does nothing if autoinstallation is disabled.

    Args:
        lazy_mod (LazyModule):
                A lazy module, like that returned by ``lazy_import``.
    """
    from polygraphy import config

    if not config.AUTOINSTALL_DEPS:
        return

    try:
        # It doesn't matter which attribute we try to get as any call to `__getattr__` will
        # trigger the automatic installation.
        getattr(lazy_mod, "__fake_polygraphy_autoinstall_attr")
    except:
        pass


def import_from_script(path, name):
    """
    Imports a specified symbol from a Python script.

    Args:
        path (str): A path to the Python script. The path must include a '.py' extension.
        name (str): The name of the symbol to import from the script.

    Returns:
        object: The loaded symbol.
    """
    from polygraphy.logger import G_LOGGER

    dir = os.path.dirname(path)
    modname = os.path.splitext(os.path.basename(path))[0]

    sys.path.insert(0, dir)

    with contextlib.ExitStack() as stack:

        def reset_sys_path():
            del sys.path[0]

        stack.callback(reset_sys_path)

        try:
            mod = importlib.import_module(modname)
            return getattr(mod, name)
        except Exception as err:
            ext = os.path.splitext(path)[1]
            err_msg = f"Could not import symbol: {name} from script: {path}"
            if ext != ".py":
                err_msg += (
                    f"\nThis could be because the extension of the file is not '.py'. Note: The extension is: {ext}"
                )
            err_msg += f"\nNote: Error was: {err}"
            err_msg += f"\nNote: sys.path was: {sys.path}"
            G_LOGGER.critical(err_msg)
