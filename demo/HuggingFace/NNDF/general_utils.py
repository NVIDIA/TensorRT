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

"""Common utils used by demo folder.
Note: 
- For now, users/developers that are contributing to TensorRT OSS should NOT import non-default Python packages in this file, because the test pipeline's boot-up process cannot load extra dependencies. In the near future, alternative solutions such as creating a separate boot-up util list can be possible. 
- Users/developers that are just using the TensorRT OSS without contributing are still free to modify this file and customize for deployment.
"""

import os
import shutil
import timeit
import math

from datetime import datetime
from shutil import rmtree
from typing import Callable, Union, List
from collections import defaultdict
from statistics import mean, median
from glob import glob

# NNDF
from NNDF.networks import NNConfig, NetworkResult, NetworkMetadata, TimingProfile
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

def process_per_result_entries(script_category: List[str], results: List[NetworkResult], max_output_char:int = 30):
    """Prints tabulations for each entry returned by the runtime result."""
    def _shorten_text(w):
        l = len(w)
        if l > max_output_char:
            return w[0:max_output_char // 2] + " ... " + w[-max_output_char//2:]
        return w

    headers = ["script", "network_part", "accuracy", "runtime", "input", "output"]
    row_data_by_input = defaultdict(list)
    for cat, result in zip(script_category, results):
        for nr in result.network_results:
            for runtime in  nr.median_runtime:
                row_data_by_input[hash(nr.input)].append([
                    cat,
                    runtime.name,
                    result.accuracy,
                    runtime.runtime,
                    _shorten_text(nr.input),
                    _shorten_text(nr.semantic_output)
                ])

    return headers, dict(row_data_by_input)

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
    stmt: Union[Callable, str], timing_profile: TimingProfile
) -> None:
    """
    Measures the time it takes to run Pythonic inference code.
    Statement given should be the actual model inference like forward() in torch.

    Args:
        stmt (Union[Callable, str]): Callable or string for generating numbers.
        timing_profile (TimingProfile): The timing profile settings with the following fields.
            warmup (int): Number of iterations to run as warm-up before actual measurement cycles.
            number (int): Number of times to call function per iteration.
            iterations (int): Number of measurement cycles.
            duration (float): Minimal duration for measurement cycles.
            percentile (int or list of ints): key percentile number(s) for measurement.
    """

    def simple_percentile(data, p):
        """
        Temporary replacement for numpy.percentile() because TRT CI/CD pipeline requires additional packages to be added at boot up in this general_utils.py file.
        """
        assert p >= 0 and p <= 100, "Percentile must be between 1 and 99"
        
        rank = len(data) * p / 100
        if rank.is_integer():
            return sorted(data)[int(rank)]
        else:
            return sorted(data)[int(math.ceil(rank)) - 1]

    warmup = timing_profile.warmup
    number = timing_profile.number
    iterations = timing_profile.iterations
    duration = timing_profile.duration
    percentile = timing_profile.percentile

    G_LOGGER.debug(
        "Measuring inference call with warmup: {} and number: {} and iterations {} and duration {} secs".format(
            warmup, number, iterations, duration
        )
    )
    # Warmup
    warmup_mintime = timeit.repeat(stmt, number=number, repeat=warmup)
    G_LOGGER.debug("Warmup times: {}".format(warmup_mintime))

    # Actual measurement cycles
    results = []
    start_time = datetime.now()
    iter_idx = 0
    while iter_idx < iterations or (datetime.now() - start_time).total_seconds() < duration:
        iter_idx += 1
        results.append(timeit.timeit(stmt, number=number))

    if isinstance(percentile, int):
        return simple_percentile(results, percentile) / number
    else:
        return [simple_percentile(results, p) / number for p in percentile]

class NNFolderWorkspace:
    """
    For keeping track of workspace folder and for cleaning them up.
    Due to potential corruption of ONNX model conversion, the workspace is split up by model variants.
    """

    def __init__(
        self, network_name: str, metadata: NetworkMetadata, working_directory: str
    ):
        self.rootdir = working_directory
        self.metadata = metadata
        self.network_name = network_name
        self.dpath = os.path.join(self.rootdir, self.network_name, metadata.variant)
        os.makedirs(self.dpath, exist_ok=True)

    def set_model_path(self, metadata_serialized, is_encoder_decoder: bool) -> str:
        '''
        Create subdirectory for models with different config(e.g. kv cache)
        '''
        self.model_path = os.path.join(self.dpath, metadata_serialized)
        self.decoder_path = os.path.join(self.model_path, "decoder")
        os.makedirs(self.decoder_path, exist_ok=True)
        if is_encoder_decoder:
            self.encoder_path = os.path.join(self.model_path, "encoder")
            os.makedirs(self.encoder_path, exist_ok=True)  
        # For decoder only models, there is no encoder
        else:
            self.encoder_path = None

        # If is kv cache mode, need to separate non kv mode and kv mode for decoder
        if self.metadata.other.kv_cache:
            self.decoder_non_kv_path = os.path.join(self.decoder_path, "non-kv")
            self.decoder_kv_path = os.path.join(self.decoder_path, "kv")
            os.makedirs(self.decoder_non_kv_path, exist_ok=True)
            os.makedirs(self.decoder_kv_path, exist_ok=True)

        return self.model_path, self.encoder_path, self.decoder_path
    
    def get_path(self) -> str:
        return self.dpath
    
    def get_model_path(self) -> str:
        return self.model_path
    
    def get_encoder_path(self) -> str:
        return self.encoder_path
    
    def get_decoder_path(self) -> str:
        return self.decoder_path
    
    def get_decoder_path_kv(self) -> (str, str):
        if not self.metadata.other.kv_cache:
            raise RuntimeError("Trying to access kv specific folder in non kv mode")
        else:
            return self.decoder_kv_path, self.decoder_non_kv_path

    def cleanup(self, force_remove: bool = False) -> None:
        '''
        Cleanup would remove all the contents in the workspace.
        '''
        if force_remove:
            return shutil.rmtree(self.dpath)
        
        if self.is_encoder_decoder_path_set:
            if self.encoder_path is not None:
                remove_if_empty(self.encoder_path)
            if self.metadata.other.kv_cache:
                remove_if_empty(
                    self.decoder_kv_path
                )
                remove_if_empty(
                    self.decoder_non_kv_path
                )
            remove_if_empty(
                self.decoder_path
            )
        
        remove_if_empty(self.model_path)
        remove_if_empty(self.dpath)
