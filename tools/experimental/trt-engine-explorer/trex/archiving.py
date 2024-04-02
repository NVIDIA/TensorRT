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
This file contains archiving functionality.
"""

__all__ = ["EngineArchive", "get_reader", "get_writer"]


import os
import io
import json
import datetime
from typing import List
from io import BytesIO
from zipfile import ZipFile, Path, ZIP_DEFLATED
import tensorrt as trt


class regular_file_writer(object):
    """Adaptor for file writer context manager"""
    def __init__(self, fname: str) -> None:
        self.fname = fname

    def __enter__(self):
        self.f = open(self.fname, 'w')
        return self

    def write(self, text:str):
        self.f.write(text)

    def __exit__(self, *args) -> None:
        self.f.close()


class regular_file_reader(object):
    """Adaptor for file reader context manager"""
    def __init__(self, fname: str) -> None:
        self.fname = fname

    def __enter__(self):
        self.f = open(self.fname, 'r')
        return self

    def read(self) -> str:
        return self.f.read()

    def readlines(self) -> List[str]:
        return self.f.readlines()

    def __exit__(self, *args) -> None:
        self.f.close()


class zip_file_writer(object):
    """Adaptor for Zip file writer context manager"""
    def __init__(self, zipf: ZipFile, fname: str) -> None:
        self.zipf = zipf
        self.fname = os.path.basename(fname)

    def __enter__(self):
        return self

    def write(self, text:str):
        self.zipf.writestr(self.fname, text)

    def __exit__(self, *args) -> None:
        pass


class zip_file_reader(object):
    """Adaptor for Zip file reader context manager"""
    def __init__(self, zipf: ZipFile, fname: str) -> None:
        self.zipf = zipf
        self.fname = os.path.basename(fname)

    def __enter__(self):
        return self

    def read(self):
        return self.zipf.read(self.fname)

    def readlines(self):
        p = Path(self.zipf, self.fname)
        txt = p.read_text()
        for line in txt.split("\n"):
            yield line

    def __exit__(self, *args) -> None:
        pass


class StreamTrtLogger(trt.ILogger):
    """Writes TRT log messages to a buffer"""
    def __init__(self, user_logger):
        trt.ILogger.__init__(self)
        self.user_logger = user_logger
        self.buffer = io.StringIO()
        self.proxy_logger = trt.Logger()

    def log(self, severity: trt.ILogger.Severity, msg: str):
        if severity <= self.user_logger.min_severity:
            self.user_logger.log(severity, msg)
        self.buffer.write(msg + "\n")


class NullTrtLogger(trt.ILogger):
    """TRT log messages blackhole"""
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity: trt.ILogger.Severity, msg: str):
        pass


class EngineArchive(object):
    """Interface to a TensorRT Engine Archive (TEA) file"""
    __version__ = "1.0"

    def __init__(self, archive_filename: str, override_files: bool=True) -> None:
        if os.path.exists(archive_filename):
            if override_files:
                os.remove(archive_filename)
            else:
                raise FileExistsError(f"TensorRT engine archive {archive_filename} exists")
        self.archive_filename = archive_filename
        self.zipf = None

    def open(self):
        self.zipf = ZipFile(self.archive_filename, 'a', compression=ZIP_DEFLATED, compresslevel=9)

    def close(self):
        assert self.zipf is not None
        self.zipf.testzip()
        self.zipf.close()
        self.zipf = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def writef_txt(self, fname: str, text: str):
        """Write a text file to the TEA"""
        with self:
            self.writer(fname).write(text)
            self.zipf.testzip()

    def writef_bin(self, fname: str, content: any):
        """Write a binary file to the TEA"""
        with self:
            self.writer(fname).write(BytesIO(content).getvalue())
            self.zipf.testzip()

    def readf(self, fname: str):
        """Read a file from the TEA"""
        with self:
            return self.reader(fname).read()

    def writer(self, fname: str):
        assert self.zipf is not None
        return zip_file_writer(self.zipf, fname)

    def reader(self, fname: str):
        assert self.zipf is not None
        return zip_file_reader(self.zipf, fname)

    def archive_build_config(self,
        config: trt.IBuilderConfig,
        build_duration: datetime.timedelta
    ):
        as_dict = lambda cfg: {attr: str(getattr(cfg, attr))
            for attr in dir(cfg) if not callable(getattr(cfg, attr)) and not attr.startswith("__")}
        build_dict = {"engine_build_duration": build_duration.total_seconds()}
        build_dict.update(as_dict(config))
        as_json = json.dumps(build_dict, ensure_ascii=False, indent=4)
        self.writef_txt("build_cfg.json", as_json)

    def archive_timing_cache(self, config: trt.IBuilderConfig):
        cache = config.get_timing_cache()
        if cache is None:
            return
        self.writef_bin("timing.cache", cache.serialize())

    def archive_plan_info(self, plan: trt.IHostMemory):
        assert plan
        with trt.Runtime(NullTrtLogger()) as runtime:
            engine = runtime.deserialize_cuda_engine(plan)
            # Explicitly remove some attributes which explode when getattr is called on them.
            bad_attrs = ["weight_streaming_budget",
                "minimum_weight_streaming_budget", "streamable_weights_size"]
            safe_attrs = [attr for attr in dir(engine)
                if attr not in bad_attrs and not callable(getattr(engine, attr)) and not attr.startswith("__")]
            plan_dict = {attr: str(getattr(engine, attr)) for attr in safe_attrs}

            bindings = plan_dict["io_tensors"] = {}
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                dtype = engine.get_tensor_dtype(name)
                shape = engine.get_tensor_shape(name)
                location = engine.get_tensor_location(name)
                mode = engine.get_tensor_mode(name)
                bindings[name] = {
                    "mode": str(mode),
                    "dtype": str(dtype),
                    "shape": str(shape),
                    "location": str(location)
                }
            as_json = json.dumps(plan_dict, ensure_ascii=False, indent=4)
            self.writef_txt("plan_cfg.json", as_json)


    class _Builder(trt.Builder):
        """A trt.Builder decorator class which adds archiving functionality."""
        def __init__(self, logger, tea):
            self.tea = tea
            self.tea_logger = StreamTrtLogger(logger)
            trt.Builder.__init__(self, self.tea_logger)

        def build_serialized_network(self,
            network: trt.INetworkDefinition,
            config: trt.IBuilderConfig
        ):
            start = datetime.datetime.now()
            plan = trt.Builder.build_serialized_network(self, network, config)
            end = datetime.datetime.now()
            build_duration = end - start
            # Save to archive
            self.tea.writef_txt("build.txt", self.tea_logger.buffer.getvalue())
            self.tea.writef_bin("engine.trt", plan)
            self.tea.archive_build_config(config, build_duration)
            self.tea.archive_plan_info(plan)
            self.tea.archive_timing_cache(config)
            return plan

    def Builder(self, logger):
        assert trt.__version__ >= "10.0.0", "This TEA functionality requires TRT 10.0+"
        return self._Builder(logger, self)


def get_writer(tea: EngineArchive, fname:str):
    if tea:
        return tea.writer(fname)
    else:
        return regular_file_writer(fname)


def get_reader(tea: EngineArchive, fname:str):
    if tea:
        return tea.reader(fname)
    else:
        return regular_file_reader(fname)
