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
from pathlib import Path

from polygraphy import mod, util
from polygraphy.mod.trt_importer import lazy_import_trt
from polygraphy.logger import G_LOGGER

trt = lazy_import_trt()

@mod.export()
def FileReader(
    filepath,
    BaseClass=None,
):
    """
    Class that supplies data to TensorRT from a stream. This may help reduce memory usage during deserialization.

    Args:
        filepath (str): 
                The path to the serialized file.

    """
    BaseClass = util.default(BaseClass, trt.IStreamReader)

    class FileReaderClass(BaseClass):
        """
        Class that supplies data to TensorRT from a stream. This may help reduce memory usage during deserialization. 
        """

        def __init__(self):
            # Must explicitly initialize parent for any trampoline class! Will mysteriously segfault without this.
            BaseClass.__init__(self)  # type: ignore

            self.filepath = filepath

            if not Path(self.filepath).exists():
                G_LOGGER.error(f"File at {self.filepath} does not exist!")

            self.mode = 'rb'
            self.file = open(self.filepath, self.mode)
            if not self.file:
                G_LOGGER.error(f"Failed to open file at {self.filepath}!")

            self.make_func = FileReader

        def read(self, size: int) -> bytes:
            return self.file.read(size)

        def seek(self, offset: int, whence: int = 0) -> int:
            """
            Seek to a position in the stream. Required for IStreamReaderV2.

            Args:
                offset: The offset to seek to
                whence: How to interpret the offset (0=absolute, 1=relative to current, 2=relative to end)

            Returns:
                The new absolute position
            """
            return self.file.seek(offset, whence)

        def free(self):
            if self.file:
                self.file.close()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.free()

        def __repr__(self):
            return util.make_repr(
                "FileReader",
                self.filepath,
                BaseClass=BaseClass,
            )[0]

    return FileReaderClass()
