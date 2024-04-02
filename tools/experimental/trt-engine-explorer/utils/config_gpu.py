#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
This file contains code to configure the GPU's working clocks.
"""


from typing import Tuple
from threading import Thread, Condition
from enum import Enum, auto
import logging
from contextlib import contextmanager
import pynvml


@contextmanager
def nvmlContext(*args, **kwds):
    gpu_id = args[0]
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    try:
        yield handle
    finally:
        pynvml.nvmlShutdown()


class GPUMonitor():
    """Monitor GPU activity"""

    def _sample_gpu_state(handle):
        assert handle
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pwr = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000
            cps = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            # Clock list: https://github.com/nicolargo/nvidia-ml-py3/blob/master/pynvml.py#L95
            sm_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            mem_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            graphics_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            print(f"pwr: {pwr} temp: {temp} util_rate.gpu={util_rate.gpu} "
                f"util_rate.memory={util_rate.memory} cnt processes={len(cps)} "
                f"Clocks: sm={sm_clock_mhz} mem={mem_clock_mhz} graphics={graphics_clock_mhz}")

            # Throttle reasons
            # https://github.com/nicolargo/nvidia-ml-py3/blob/master/pynvml.py#L570
            # https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksThrottleReasons.html
            tr = (pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))
            # GPU is idle
            tr_idle = tr & pynvml.nvmlClocksThrottleReasonGpuIdle
            # GPU clocks are limited by current setting of applications clocks
            tr_appsettings = tr & pynvml.nvmlClocksThrottleReasonApplicationsClocksSetting
            # SW Power Scaling algorithm is reducing the clocks below requested clocks
            tr_sw_power = tr & pynvml.nvmlClocksThrottleReasonSwPowerCap
            # HW Power Brake Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
            # This is an indicator of: External Power Brake Assertion being triggered (e.g. by the system power supply)
            tr_hw_slowdown = tr & pynvml.nvmlClocksThrottleReasonHwSlowdown
            if tr:
                print(f"Throttling = idle={tr_idle} app={tr_appsettings} "
                    f"power={tr_sw_power} hardware={tr_hw_slowdown}")
        except pynvml.nvml.NVMLError as e:
            logging.warning(f"Could not read GPU state")

    def gpu_monitor(self):
        self.done_cond.acquire()
        with nvmlContext(self.gpu_id) as handle:
            while not self.is_done:
                self._sample_gpu_state(handle)
                self.done_cond.wait(self.sampling_interval)
        self.done_cond.release()

    def __init__(self, enabled: bool, gpu_id: int=0, sampling_interval: float=1.):
        self.enabled = enabled
        if not enabled:
            return
        self.gpu_id = gpu_id
        self.sampling_interval = sampling_interval
        self.is_done = False
        self.monitor = Thread(target=self.gpu_monitor)
        self.done_cond = Condition()

    def __enter__(self):
        if not self.enabled:
            return
        self.monitor.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.enabled:
            return
        self.done_cond.acquire()
        self.is_done = True
        self.done_cond.notify()
        self.done_cond.release()
        self.monitor.join()


class GPUConfigurator():
    """GPU configuration interface"""
    class Key(Enum):
        INIT_POWER_LIMIT = auto()
        MIN_POWER_LIMIT = auto()
        MAX_POWER_LIMIT = auto()
        MEMORY_CLOCK = auto()
        COMPUTE_CLOCK = auto()

    def __init__(
        self,
        power_limit: float=None,
        compute_clk: int=None,
        memory_clk: int=None,
        gpu_id: int=0,
        dont_lock_clocks: bool=False
    ):
        self.power_limit = power_limit
        self.compute_clk = compute_clk
        self.memory_clk = memory_clk
        self.gpu_id = gpu_id
        self.dont_lock_clocks = dont_lock_clocks
        self.set_power_limit = power_limit is not None
        self.set_lock = (compute_clk is not None and memory_clk is not None) and (not dont_lock_clocks)
        if self.set_power_limit:
            with nvmlContext(self.gpu_id) as handle:
                self.power_readings_stats = self._extract_power_limits(handle)

    def __enter__(self):
        with nvmlContext(self.gpu_id) as handle:
            if self.set_power_limit:
                self._set_power_limit(handle, self.power_limit)
            if self.set_lock:
                self._lock_clocks(handle)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        with nvmlContext(self.gpu_id) as handle:
            if self.set_power_limit:
                init_power_limit = self.power_readings_stats[self.Key.INIT_POWER_LIMIT]
                self._set_power_limit(handle, init_power_limit)
            if self.set_lock:
                self._unlock_clocks(handle)

    # Helper functions
    def _extract_power_limits(self, handle):
        def to_watt(power_milliwatt: int):
            return power_milliwatt // 1000

        min_power_limit, max_power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        try:
            cur_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        except pynvml.nvml.NVMLError as e:
            logging.warning(f"Could read power limit constraints ({e}).")
            cur_power_limit = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)

        # Power limit stats in Watts.
        power_readings_stats = {
            self.Key.INIT_POWER_LIMIT: to_watt(cur_power_limit),
            self.Key.MIN_POWER_LIMIT: to_watt(min_power_limit),
            self.Key.MAX_POWER_LIMIT: to_watt(max_power_limit),
        }
        return power_readings_stats

    def _set_power_limit(self, handle, power_limit: float):
        def to_milliwatt(power_watt: int):
            return power_watt * 1000

        min_power_limit = self.power_readings_stats[self.Key.MIN_POWER_LIMIT]
        max_power_limit = self.power_readings_stats[self.Key.MAX_POWER_LIMIT]
        power_limit = min(max_power_limit, max(min_power_limit, power_limit))
        logging.warning(f"Setting power limit to {power_limit} Watts")
        try:
            power_limit = to_milliwatt(power_limit)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, power_limit)
        except pynvml.nvml.NVMLError_InvalidArgument as e:
            self.set_power_limit = False
            logging.warning(f"Could not set power limits ({e})\n"
            f"\twhile using power_limit = {power_limit} Watts\n"
            "\tTry different power limit.")
        except pynvml.nvml.NVMLError as e:
            self.set_power_limit = False
            logging.warning(f"Could not set power limits ({e}).")


    def _lock_clocks(self, handle):
        try:
            pynvml.nvmlDeviceSetApplicationsClocks(handle, self.memory_clk, self.compute_clk)
            pynvml.nvmlDeviceSetGpuLockedClocks(handle,
                minGpuClockMHz=self.compute_clk,
                maxGpuClockMHz=self.compute_clk)
            logging.warning(f"Set max memory clock = {self.memory_clk} MHz")
            logging.warning(f"Set max compute clock = {self.compute_clk} MHz")
            logging.warning(f"Locked graphics clock = {self.compute_clk} MHz")
        except pynvml.nvml.NVMLError_NoPermission as e:
            logging.warning(f"Could not lock clocks ({e}).\n"
            "\tTry running as root or locking the clocks from the commandline:\n"
            f"\t\tsudo nvidia-smi --lock-gpu-clocks={self.compute_clk},{self.compute_clk}\n"
            f"\t\tsudo nvidia-smi --applications-clocks={self.memory_clk},{self.compute_clk}")
        except pynvml.nvml.NVMLError_InvalidArgument as e:
            logging.warning(f"Could not lock clocks ({e})\n"
            f"\twhile using memory clock = {self.memory_clk} MHz\n"
            f"\tand using compute clock = {self.compute_clk} MHz\n"
            "\tTry different clocks frequencies.")
        except pynvml.nvml.NVMLError as e:
            logging.warning(f"Could not lock clocks ({e}).")

    def _unlock_clocks(self, handle):
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
            pynvml.nvmlDeviceResetApplicationsClocks(handle)
            logging.warning(f"Unlocked device clocks.")
        except pynvml.nvml.NVMLError as e:
            logging.warning(f"Could not unlock clocks ({e}).\n"
            "\tTry running as root or unlocking the clocks from the commandline:\n"
            "\t\tsudo nvidia-smi --reset-gpu-clocks\n"
            "\t\tsudo nvidia-smi --reset-applications-clocks")


def get_max_clocks(dev: int) -> Tuple[int, int]:
    with nvmlContext(dev) as handle:
        mem_clks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
        max_mem_clk = mem_clks[0]
        gr_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, max_mem_clk)
        max_gr_clk = gr_clocks[0]
        return max_mem_clk, max_gr_clk
