#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Holds logic for modifying and removing invalid CUDA libraries in LD_LIBRARY_PATH.

Users may have CUDA libraries in LD_LIBRARY_PATH which causes issues with Torch cublas.
This problem only occurs on Linux.
See:
    https://github.com/pytorch/pytorch/issues/94294
    https://github.com/pytorch/pytorch/issues/64097
"""

import os
import sys
import glob
import shutil

import subprocess as sp
from NNDF.logger import G_LOGGER

def bootstrap_ld_library_path() -> bool:
    """
    Modifies the LD_LIBRARY_PATH if applicable and then spawns a child process
    using first "poetry" and then "python3"/"python" if "poetry" fails.
    """
    if os.environ.get("TRT_OSS_DISABLE_BOOTSTRAP") or "linux" not in sys.platform:
        return False

    # Walk through each path in environment to see if there are cublas libraries being loaded.
    paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    new_paths = []
    modified_path = False
    for path in paths:
        for lib in ("cublas", "cudart", "cublasLt"):
            g = glob.glob(os.path.join(path, f"lib{lib}.so.*"))
            if g:
                modified_path = True
                G_LOGGER.warning(f"Discarding `{path}` from LD_LIBRARY_PATH since it contains CUDA libraries.")
                break
            else:
                new_paths.append(path)


    if not modified_path:
        return False
    else:
        warning_msg = ("Attempting to bootstrap altered LD_LIBRARY_PATH. "
                       "\nYou can disable this with TRT_OSS_DISABLE_BOOTSTRAP=1 however frameworks performance may be impacted. "
                       "\nThere are known issues with cuBLAS loading and PyTorch compatability "
                       "that is still being resolved for most CUDA <= 12.1 and Torch setups. See: "
                       "\n   - https://github.com/pytorch/pytorch/issues/94294"
                       "\n   - https://github.com/pytorch/pytorch/issues/64097\n")
        G_LOGGER.warning(warning_msg)

    G_LOGGER.info(f"CUDA detected in path. Restarting scripts with modified LD_LIBRARY_PATH: {new_paths}")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new_paths)
    # To prevent potential recursion, we add one more modification just in case.
    os.environ["TRT_OSS_DISABLE_BOOTSTRAP"] = "1"

    # Spawn a new child process instead.
    try:
        # Use the same python exe that invoked this script
        default_python = sys.executable

        # Demo supports both poetry and python3 invocation.
        # Check if poetry works first.
        cmd = [default_python] + list(sys.argv)
        if shutil.which("poetry") is not None:
            poetry_cmd = ["poetry", "run"] + cmd

            # Poetry command will be tried. If it fails, we ignore the error and fallback to default python.
            try:
                # Instantiate a secondary child process.
                sp.check_call(" ".join(poetry_cmd), env=dict(os.environ), cwd=os.getcwd(), shell=True)
                return True
            except:
                pass

        # Default python fallback.
        sp.check_call(" ".join(cmd), env=dict(os.environ), cwd=os.getcwd(), shell=True)
    except Exception as e:
        G_LOGGER.error("Unable to start a new process with modified LD_LIBRARY_PATH. Consider removing CUDA lib in LD_LIBRARY_PATH manually.")
        G_LOGGER.error(str(e))
        G_LOGGER.warning("Attempting to continue with demo.")

    return True
