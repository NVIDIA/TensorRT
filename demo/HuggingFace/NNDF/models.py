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

"""
File for containing model file abstraction. Useful for generating models.
"""

import os
from abc import ABCMeta, abstractmethod
from typing import Union, List
from shutil import copytree, rmtree

# polygraphy
from polygraphy.backend.trt import (
    network_from_onnx_path,
    engine_from_network,
    save_engine,
    Profile,
)

from polygraphy.backend.trt import CreateConfig
from polygraphy.logger import G_LOGGER as PG_LOGGER

# torch
from torch import load, save
from torch.nn import Module

# tensorrt
from tensorrt import PreviewFeature, MemoryPoolType

# TRT-HuggingFace
from NNDF.networks import NetworkMetadata
from NNDF.logger import G_LOGGER


class ModelFileConverter:
    """Abstract class for converting one model format to another."""

    def __init__(self, onnx_class, torch_class, trt_engine_class):
        self.onnx_class = onnx_class
        self.torch_class = torch_class
        self.trt_engine_class = trt_engine_class

    def torch_to_onnx(
        self, output_fpath: str, model: Module, network_metadata: NetworkMetadata
    ):
        """
        Converts a torch.Model into an ONNX model on disk specified at output_fpath.

        Arg:
            output_fpath (str): File location of the generated ONNX file.
            input_fpath (str): Input file location of the generated ONNX file.
            network_metadata (NetworkMetadata): Network metadata of the network being converted.

        Returns:
            ONNXModelFile: Newly generated ONNXModelFile
        """
        raise NotImplementedError(
            "Current model does not support exporting to ONNX model."
        )

    def onnx_to_torch(
        self, output_fpath: str, input_fpath: str, network_metadata: NetworkMetadata
    ):
        """
        Converts ONNX file into torch.Model which is written to disk.

        Arg:
            output_fpath (str): File location of the generated ONNX file.
            input_fpath (str): Input file location of the generated ONNX file.
            network_metadata (NetworkMetadata): Network metadata of the network being converted.

        Returns:
            TorchModelFile: Newly generated TorchModelFile
        """
        raise NotImplementedError(
            "Current model does not support exporting to torch model."
        )

    def onnx_to_trt(
        self,
        output_fpath: str,
        input_fpath: str,
        network_metadata: NetworkMetadata,
        profiles: List[Profile],
        preview_features: List[PreviewFeature],
    ):
        """
        Converts ONNX file to TRT engine.
        Since TensorRT already supplies converter functions and scripts,
        a default implementation is already provided.

        Arg:
            output_fpath (str): File location of the generated ONNX file.
            input_fpath (str): Input file location of the generated ONNX file.
            network_metadata (NetworkMetadata): Network metadata of the network being converted.
            profiles (List[polygraphy.backend.trt.Profile]): The optimization profiles used to build the engine.
            preview_features (List[tensorrt.PreviewFeature]): The preview features to enable when building the engine.

        Returns:
            TRTEngineFile: Newly generated engine.
        """
        result = self.trt_engine_class(output_fpath, network_metadata)

        G_LOGGER.info("Using optimization profiles: {:}".format(profiles))

        try:
            self.trt_inference_config = CreateConfig(
                tf32=True,
                fp16=network_metadata.precision.fp16,
                memory_pool_limits = {MemoryPoolType.WORKSPACE: result.max_trt_workspace * 1024 * 1024},
                profiles=profiles,
                precision_constraints=("obey" if result.use_obey_precision_constraints() else None),
                preview_features=preview_features
            )
        except TypeError as e:
            G_LOGGER.error(f"This demo may have an outdated polygraphy. Please see requirements.txt for more details.")
            raise e

        if G_LOGGER.level == G_LOGGER.DEBUG:
            g_logger_verbosity = PG_LOGGER.EXTRA_VERBOSE
        elif G_LOGGER.level == G_LOGGER.INFO:
            g_logger_verbosity = PG_LOGGER.INFO
        else:
            g_logger_verbosity = PG_LOGGER.WARNING

        with PG_LOGGER.verbosity(g_logger_verbosity):
            network_definition = result.get_network_definition(network_from_onnx_path(input_fpath))

            trt_engine = engine_from_network(
                network_definition, config=self.trt_inference_config
            )
            save_engine(trt_engine, output_fpath)

        return result


class NNModelFile(metaclass=ABCMeta):
    """
    Model abstraction. Allows for loading model as various formats.
    The class assumes models live on the disk in order to reduce complexity of model loading into memory.
    The class guarantees that once export functions are called, models exist on the disk for other
    code to parse or use in other libraries.
    """

    def __init__(
        self,
        default_converter: ModelFileConverter = None,
        network_metadata: NetworkMetadata = None,
    ):
        """
        Since torch functions often allow for models to either be from disk as fpath or from a loaded object,
        we provide a similar option here. Arguments can either be a path on disk or from model itself.

        Args:
            model (Union[str, torch.Model]): Location of the model as fpath OR loaded torch.Model object.
        """
        if default_converter is not None:
            self.default_converter = default_converter()
        else:
            self.default_converter = NullConverter()

        self.network_metadata = network_metadata

    def as_torch_model(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Converts ONNX file into torch.Model which is written to disk.
        Uses provided converter to convert object or default_convert is used instead if available.

        Arg:
            output_fpath (str): File location of the generated torch file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.

        Returns:
            TorchModelFile: Newly generated TorchModelFile
        """
        raise NotImplementedError(
            "Current model does not support exporting to pytorch model."
        )

    def as_onnx_model(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Converts current model into an ONNX model.
        Uses provided converter to convert object or default_convert is used instead if available.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.

        Returns:
            ONNXModelFile: Newly generated ONNXModelFile
        """
        raise NotImplementedError(
            "Current model does not support exporting to onnx model."
        )

    def as_trt_engine(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
        profiles: List[Profile] = [],
        preview_features: List[PreviewFeature] = []
    ):
        """
        Converts current model into an TRT engine.
        Uses provided converter to convert object or default_convert is used instead if available.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.
            profiles (List[polygraphy.backend.trt.Profile]): The optimization profiles used to build the engine.
            preview_features (List[tensorrt.PreviewFeature]): The preview features to enable when building the engine.

        Returns:
            TRTEngineFile: Newly generated ONNXModelFile
        """
        raise NotImplementedError(
            "Current model does not support exporting to trt engine."
        )

    @abstractmethod
    def cleanup(self) -> None:
        """Cleans up any saved models or loaded models from memory."""


class TorchModelFile(NNModelFile):
    def __init__(
        self,
        model: Union[str, Module],
        default_converter: ModelFileConverter = None,
        network_metadata: NetworkMetadata = None,
    ):
        """
        Since torch functions often allow for models to either be from disk as fpath or from a loaded object,
        we provide a similar option here. Arguments can either be a path on disk or from model itself.

        Args:
            model (Union[str, torch.Model]): Location of the model as fpath OR loaded torch.Model object.
        """
        super().__init__(default_converter, network_metadata)

        if isinstance(model, Module):
            self.is_loaded = True
            self.fpath = None
            self.model = model
        else:
            self.is_loaded = False
            self.fpath = model
            self.model = None

    def load_model(self) -> Module:
        """
        Loads the model from disk if isn't already loaded.
        Does not attempt to load if given model is already loaded and instead returns original instance.
        Use as_torch_model() instead to always guarantee a new instance and location on disk.

        Args:
            None

        Returns:
            torch.Model: Loaded torch model.
        """
        if self.is_loaded:
            return self.model

        return load(self.fpath)

    def as_onnx_model(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Converts the torch model into an onnx model.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.
        Return:
            (converter.onnx_class): Returns a converted instance of ONNXModelFile.
        """
        converter = self.default_converter if converter is None else converter()
        if not force_overwrite and os.path.exists(output_fpath):
            return converter.onnx_class(output_fpath, self.network_metadata)

        return converter.torch_to_onnx(
            output_fpath, self.load_model(), self.network_metadata
        )

    def as_torch_model(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Since the model is already a torch model, forces a save to specified folder and returns new TorchModelFile object from that file location.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.
        Return:
            (converter.torch_class): Returns a converted instance of TorchModelFile.
        """
        converter = self.default_converter if converter is None else converter()
        if not force_overwrite and os.path.exists(output_fpath):
            return converter.torch_class(output_fpath, self.network_metadata)

        if self.is_loaded:
            save(self.model, output_fpath)
        else:
            copytree(self.fpath, output_fpath)

        return converter.torch_class(output_fpath, self.network_metadata)

    def cleanup(self) -> None:
        if self.model:
            G_LOGGER.debug("Freeing model from memory: {}".format(self.model))
            del self.model

        if self.fpath:
            G_LOGGER.debug("Removing saved torch model from location: {}".format(self.fpath))
            rmtree(self.fpath)


class ONNXModelFile(NNModelFile):
    def __init__(
        self,
        model: str,
        default_converter: ModelFileConverter = None,
        network_metadata: NetworkMetadata = None,
    ):
        """
        Keeps track of ONNX model file. Does not support loading into memory. Only reads and writes to disk.

        Args:
            model (str): Location of the model as fpath OR loaded torch.Model object.
        """
        super().__init__(default_converter, network_metadata)
        self.fpath = model

    def as_onnx_model(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Since the model is already a onnx model, forces a save to specified folder and returns new ONNXModelFile object from that file location.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.

        Return:
            (converter.onnx_class): Returns a converted instance of ONNXModelFile.
        """
        converter = self.default_converter if converter is None else converter()
        if not force_overwrite and os.path.exists(output_fpath):
            return converter.onnx_class(output_fpath, self.network_metadata)
        else:
            copytree(self.fpath, output_fpath)

        return converter.onnx_class(output_fpath, self.network_metadata)

    def as_torch_model(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
    ):
        """
        Converts the onnx model into an torch model.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.
        Return:
            (converter.torch_class): Returns a converted instance of TorchModelFile.
        """
        converter = self.default_converter if converter is None else converter()
        if not force_overwrite and os.path.exists(output_fpath):
            return converter.torch_class(output_fpath, self.network_metadata)

        return converter.onnx_to_torch(output_fpath, self.fpath, self.network_metadata)
    
    def _cleanup_onnx_folder(self, folder_dir):
        for d in os.listdir(folder_dir):
            fpath = os.path.join(folder_dir, d)
            # Remove everything related to onnx other than engine
            if (os.path.isfile(fpath)) and (".engine" not in d):
                os.remove(fpath)

    def cleanup(self) -> None:
        G_LOGGER.debug("Removing saved ONNX model from location: {}".format(self.fpath))
        if (not self.network_metadata.other.kv_cache) or ("encoder" in self.fpath):
            # Clean up any onnx external files by removing integer named values and weight files
            workspace_path = os.path.split(self.fpath)[0]
            self._cleanup_onnx_folder(workspace_path)

        else:
            # In kv cache mode, hard to remove the decoder. Therefore need to search for temporary WAR.
            decoder_path = os.path.split(self.fpath)[0]
            decoder_non_kv_path = os.path.join(decoder_path, "non-kv")
            decoder_kv_path = os.path.join(decoder_path, "kv")
            # Remove kv and nonkv folder correspondingly.
            self._cleanup_onnx_folder(decoder_non_kv_path)
            self._cleanup_onnx_folder(decoder_kv_path)

    def as_trt_engine(
        self,
        output_fpath: str,
        converter: ModelFileConverter = None,
        force_overwrite: bool = False,
        profiles = [],
        preview_features = []
    ):
        """
        Converts the onnx model into an trt engine.

        Args:
            output_fpath (str): File location of the generated ONNX file.
            converter (ModelFileConverter): Class to convert current model instance into another.
            force_overwrite (bool): If the file already exists, tell whether or not to overwrite.
                                    Since torch models folders, can potentially erase entire folders.
            profiles (List[polygraphy.backend.trt.Profile]): The optimization profiles used to build the engine.
            preview_features (List[tensorrt.PreviewFeature]): The preview features to enable when building the engine.
        Return:
            (converter.trt_engine_class): Returns a converted instance of TRTEngineFile.
        """
        converter = self.default_converter if converter is None else converter()

        # TODO: Need to check if the old engine file is compatible with current setting
        if not force_overwrite and os.path.exists(output_fpath):
            return converter.trt_engine_class(output_fpath, self.network_metadata)

        return converter.onnx_to_trt(
            output_fpath,
            self.fpath,
            self.network_metadata,
            profiles,
            preview_features
        )


class TRTEngineFile(NNModelFile):

    @abstractmethod
    def use_obey_precision_constraints(self):
        pass

    # get_network_definition can be overloaded to alter the network definition.
    # For example, this function can be used to change the precisions of ops or
    # data type of intermediate tensors.
    def get_network_definition(self, network_definition):
        return network_definition

    def __init__(
        self,
        model: str,
        default_converter: ModelFileConverter = None,
        network_metadata: NetworkMetadata = None,
    ):
        super().__init__(default_converter, network_metadata)
        self.fpath = model
        self.max_trt_workspace = 3072

    def cleanup(self) -> None:
        G_LOGGER.debug("Removing saved engine model from location: {}".format(self.fpath))
        os.remove(self.fpath)


class NullConverter(ModelFileConverter):
    def __init__(self):
        super().__init__(ONNXModelFile, TorchModelFile, TRTEngineFile)
