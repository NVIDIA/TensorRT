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

"""Common utils used by demo folder."""

import os
import shutil
import timeit

from shutil import rmtree
from typing import Callable, Union, List
from collections import defaultdict
from statistics import mean, median
from glob import glob

# NNDF
from NNDF.networks import NNConfig, NetworkResult, NetworkMetadata
from NNDF.logger import G_LOGGER

# Used for HuggingFace setting random seed
RANDOM_SEED = 42

# Networks #
def register_network_folders(
    root_dir: str, config_file_str: str = "*Config.py"
) -> List[str]:
    networks = []
    for network_configs in glob(os.path.join(root_dir, "*", config_file_str)):
        network_name = os.path.split(os.path.split(network_configs)[0])[1]
        networks.append(network_name)
    return networks


def process_results(category: List[str], results: List[NetworkResult], nconfig: NNConfig):
    """
    Calculate and process results across multiple runs.
    """
    general_stats = ["script", "accuracy"]
    runtime_result_row_names = list(nconfig.NETWORK_SEGMENTS)
    if nconfig.NETWORK_FULL_NAME not in nconfig.NETWORK_SEGMENTS:
        runtime_result_row_names.append(nconfig.NETWORK_FULL_NAME)

    rows = []
    row_entry = []
    for cat, result in zip(category, results):
        # Process runtime results for each group
        runtime_results = defaultdict(list)
        for runtimes in [nr.median_runtime for nr in result.network_results]:
            for runtime in runtimes:
                runtime_results[runtime.name].append(runtime.runtime)

        # Calculate average runtime for each group
        average_group_runtime = {k: mean(v) for k, v in runtime_results.items()}
        row_entry = [cat, result.accuracy] + [
            average_group_runtime[n] for n in runtime_result_row_names
        ]
        rows.append(row_entry)

    headers = general_stats + [r + " (sec)" for r in runtime_result_row_names]
    return headers, rows


# IO #
def confirm_folder_delete(
    fpath: str, prompt: str = "Confirm you want to delete entire folder?"
) -> None:
    """
    Confirms whether or not user wants to delete given folder path.

    Args:
        fpath (str): Path to folder.
        prompt (str): Prompt to display

    Returns:
        None
    """
    msg = prompt + " {} [Y/n] ".format(fpath)
    confirm = input(msg)
    if confirm == "Y":
        rmtree(fpath)
    else:
        G_LOGGER.info("Skipping file removal.")


def remove_if_empty(
    fpath: str,
    success_msg: str = "Folder successfully removed.",
    error_msg: str = "Folder cannot be removed, there are files.",
) -> None:
    """
    Removes an entire folder if folder is empty. Provides print info statements.

    Args:
        fpath: Location to folder
        success_msg: Success message.
        error_msg: Error message.

    Returns:
        None
    """
    if len(os.listdir(fpath)) == 0:
        os.rmdir(fpath)
        G_LOGGER.info(success_msg + " {}".format(fpath))
    else:
        G_LOGGER.info(error_msg + " {}".format(fpath))


def measure_python_inference_code(
    stmt: Union[Callable, str], warmup: int = 3, number: int = 10, iterations: int = 10
) -> None:
    """
    Measures the time it takes to run Pythonic inference code.
    Statement given should be the actual model inference like forward() in torch.

    See timeit for more details on how stmt works.

    Args:
        stmt (Union[Callable, str]): Callable or string for generating numbers.
        number (int): Number of times to call function per iteration.
        iterations (int): Number of measurement cycles.
    """
    G_LOGGER.debug(
        "Measuring inference call with warmup: {} and number: {} and iterations {}".format(
            warmup, number, iterations
        )
    )
    # Warmup
    warmup_mintime = timeit.repeat(stmt, number=number, repeat=warmup)
    G_LOGGER.debug("Warmup times: {}".format(warmup_mintime))

    return median(timeit.repeat(stmt, number=number, repeat=iterations)) / number

class NNFolderWorkspace:
    """For keeping track of workspace folder and for cleaning them up."""

    def __init__(
        self, network_name: str, metadata: NetworkMetadata, working_directory: str
    ):
        self.rootdir = working_directory
        self.metadata = metadata
        self.network_name = network_name
        self.dpath = os.path.join(self.rootdir, self.network_name)
        os.makedirs(self.dpath, exist_ok=True)

    def get_path(self) -> str:
        dpath = os.path.join(self.rootdir, self.network_name)
        return dpath

    def cleanup(self, force_remove: bool = False) -> None:
        fpath = self.get_path()
        if force_remove:
            return shutil.rmtree(fpath)
        remove_if_empty(
            fpath,
            success_msg="Sucessfully removed workspace.",
            error_msg="Unable to remove workspace.",
        )
