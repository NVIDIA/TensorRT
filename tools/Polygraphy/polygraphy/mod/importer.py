#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

from polygraphy.mod import util as mod_util

# Tracks all of Polygraphy's lazy imports, excluding internal ones.
_all_external_lazy_imports = set()

# Sometimes the Python package name differs from the module name.
_PKG_NAME_FROM_MODULE = {
    "tensorrt": "nvidia-tensorrt",
}

# Some packages need additional flags to install correctly.
_EXTRA_FLAGS_FOR_MODULE = {
    "tensorrt": ["--extra-index-url=https://pypi.ngc.nvidia.com"],
    "onnx_graphsurgeon": ["--extra-index-url=https://pypi.ngc.nvidia.com"],
}


LATEST_VERSION = "latest"
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


def lazy_import(name, log=True, version=None):
    """
    Lazily import a module.

    If the POLYGRAPHY_AUTOINSTALL_DEPS environment variable is set to 1,
    missing modules are automatically installed, and existing modules may be
    upgraded if newer versions are required.

    Args:
        name (str):
                The name of the module.
        log (bool):
                Whether to log information about the module.
        version (str):
                The preferred version of the package, formatted as a version string.
                For example, ``'>=0.5.0'`` or ``'==1.8.0'``. Use ``LATEST_VERSION`` to
                indicate that the latest version of the package is preferred.

    Returns:
        LazyModule:
                A lazily loaded module. When an attribute is first accessed,
                the module will be imported.
    """
    assert (
        version is None or version == LATEST_VERSION or any(version.startswith(char) for char in ["=", ">", "<"])
    ), "version must be formatted as a version string!"

    if "polygraphy" not in name:
        _all_external_lazy_imports.add(name)

    def import_mod():
        from polygraphy import config
        from polygraphy.logger import G_LOGGER, LogMode

        def install_mod(raise_error=True):
            modname = name.split(".")[0]
            pkg = _PKG_NAME_FROM_MODULE.get(modname, modname)
            extra_flags = _EXTRA_FLAGS_FOR_MODULE.get(modname, [])

            if version == LATEST_VERSION:
                extra_flags.append("--upgrade")
            elif version is not None:
                pkg += version

            cmd = config.INSTALL_CMD + [pkg] + extra_flags
            G_LOGGER.info("Running installation command: {:}".format(" ".join(cmd)))
            status = sp.run(cmd)
            if status.returncode != 0:
                log_func = G_LOGGER.critical if raise_error else G_LOGGER.warning
                log_func(
                    "Could not automatically install required module: {:}. Please install it manually.".format(pkg)
                )

            mod = importlib.import_module(name)
            return mod

        mod = None
        try:
            mod = importlib.import_module(name)
        except ImportError:
            if config.AUTOINSTALL_DEPS:
                G_LOGGER.info("Module: '{:}' is required, but not installed. Attempting to install now.".format(name))
                mod = install_mod()
            else:
                G_LOGGER.critical(
                    "Module: '{:}' is required but could not be imported.\n"
                    "You can try setting POLYGRAPHY_AUTOINSTALL_DEPS=1 in your environment variables "
                    "to allow Polygraphy to automatically install missing modules.\n"
                    "Note that this may cause existing modules to be overwritten - hence, it may be "
                    "desirable to use a Python virtual environment or container. ".format(name)
                )

        # Auto-upgrade if necessary
        if version is not None and hasattr(mod, "__version__") and not _version_ok(mod.__version__, version):
            if config.AUTOINSTALL_DEPS:
                G_LOGGER.info(
                    "Note: Module: '{name}' version '{cur_ver}' is installed, but version '{rec_ver}' is recommended.\n"
                    "Attempting to upgrade now.".format(name=name, cur_ver=mod.__version__, rec_ver=version)
                )
                mod = install_mod(raise_error=False)  # We can try to use the other version if install fails.
            elif version != LATEST_VERSION:
                G_LOGGER.warning(
                    "Module: '{name}' version '{cur_ver}' is installed, but version '{rec_ver}' is recommended.\n"
                    "Consider installing the recommended version or setting POLYGRAPHY_AUTOINSTALL_DEPS=1 in your "
                    "environment variables to do so automatically. ".format(
                        name=name, cur_ver=mod.__version__, rec_ver=version
                    ),
                    mode=LogMode.ONCE,
                )

        if log:
            G_LOGGER.module_info(mod)

        return mod

    class LazyModule(object):
        def __getattr__(self, name):
            self = import_mod()
            return getattr(self, name)

        def __setattr__(self, name, value):
            self = import_mod()
            return setattr(self, name, value)

    return LazyModule()


def has_mod(lazy_mod, with_attr="__version__"):
    """
    Checks whether a module is available.

    Args:
        lazy_mod (LazyModule):
                A lazy module, like that returned by ``lazy_import``.
        with_attr (str):
                The name of an attribute to check for.
                This helps distinguish mock modules from real ones.
    """
    try:
        getattr(lazy_mod, with_attr)
    except:
        return False
    return True


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
            err_msg = "Could not import symbol: {:} from script: {:}".format(name, path)
            if ext != ".py":
                err_msg += "\nThis could be because the extension of the file is not '.py'. Note: The extension is: {:}".format(
                    ext
                )
            err_msg += "\nNote: Error was: {:}".format(err)
            G_LOGGER.critical(err_msg)
