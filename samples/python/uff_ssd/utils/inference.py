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

import os
import sys
import time

import tensorrt as trt
import tensorflow as tf
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import utils.engine as engine_utils # TRT Engine creation/save/load utils
import utils.model as model_utils # UFF conversion uttils

# ../../common.py
sys.path.insert(1,
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir,
        os.pardir
    )
)
import common


# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInference(object):
    """Manages TensorRT objects for model inference."""
    def __init__(self, trt_engine_path, uff_model_path, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1):
        """Initializes TensorRT objects needed for model inference.

        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
            uff_model_path (str): path of .uff model
            trt_engine_datatype (trt.DataType):
                requested precision of TensorRT engine used for inference
            batch_size (int): batch size for which engine
                should be optimized for
        """

        # We first load all custom plugins shipped with TensorRT,
        # some of them will be needed during inference
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None

        # Display requested engine settings to stdout
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))

        # If engine is not cached, we need to build it
        if not os.path.exists(trt_engine_path):
            # This function uses supplied .uff file
            # alongside with UffParser to build TensorRT
            # engine. For more details, check implmentation
            self.trt_engine = engine_utils.build_engine(
                uff_model_path, TRT_LOGGER,
                trt_engine_datatype=trt_engine_datatype,
                batch_size=batch_size)
            # Save the engine to file
            engine_utils.save_engine(self.trt_engine, trt_engine_path)

        # If we get here, the file with engine exists, so we can load it
        if not self.trt_engine:
            print("Loading cached TensorRT engine from {}".format(
                trt_engine_path))
            self.trt_engine = engine_utils.load_engine(
                self.trt_runtime, trt_engine_path)

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = \
            engine_utils.allocate_buffers(self.trt_engine)

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        input_volume = trt.volume(model_utils.ModelData.INPUT_SHAPE)
        self.numpy_array = np.zeros((self.trt_engine.max_batch_size, input_volume))

    def infer(self, image_path):
        """Infers model on given image.

        Args:
            image_path (str): image to run object detection model on
        """

        # Load image into CPU
        img = self._load_img(image_path)

        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, img.ravel())

        # When infering on single image, we measure inference
        # time to output it to the user
        inference_start_time = time.time()

        # Fetch output from the model
        [detection_out, keepCount_out] = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)

        # Output inference time
        print("TensorRT inference time: {} ms".format(
            int(round((time.time() - inference_start_time) * 1000))))

        # And return results
        return detection_out, keepCount_out

    def infer_batch(self, image_paths):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """

        # Verify if the supplied batch size is not too big
        max_batch_size = self.trt_engine.max_batch_size
        actual_batch_size = len(image_paths)
        if actual_batch_size > max_batch_size:
            raise ValueError(
                "image_paths list bigger ({}) than engine max batch size ({})".format(actual_batch_size, max_batch_size))

        # Load all images to CPU...
        imgs = self._load_imgs(image_paths)
        # ...copy them into appropriate place into memory...
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, imgs.ravel())

        # ...fetch model outputs...
        [detection_out, keep_count_out] = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=max_batch_size)
        # ...and return results.
        return detection_out, keep_count_out

    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image).reshape(
            (im_height, im_width, model_utils.ModelData.get_input_channels())
        ).astype(np.uint8)

    def _load_imgs(self, image_paths):
        batch_size = self.trt_engine.max_batch_size
        for idx, image_path in enumerate(image_paths):
            img_np = self._load_img(image_path)
            self.numpy_array[idx] = img_np
        return self.numpy_array


    def _load_img(self, image_path):
        image = Image.open(image_path)
        model_input_width = model_utils.ModelData.get_input_width()
        model_input_height = model_utils.ModelData.get_input_height()
        # Note: Bilinear interpolation used by Pillow is a little bit
        # different than the one used by Tensorflow, so if network receives
        # an image that is not 300x300, the network output may differ
        # from the one output by Tensorflow
        image_resized = image.resize(
            size=(model_input_width, model_input_height),
            resample=Image.BILINEAR
        )
        img_np = self._load_image_into_numpy_array(image_resized)
        # HWC -> CHW
        img_np = img_np.transpose((2, 0, 1))
        # Normalize to [-1.0, 1.0] interval (expected by model)
        img_np = (2.0 / 255.0) * img_np - 1.0
        img_np = img_np.ravel()
        return img_np


# This class is similar as TRTInference inference, but it manages Tensorflow
class TensorflowInference(object):
    def __init__(self, pb_model_path):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(pb_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.sess = tf.Session(graph=self.detection_graph)

    def infer(self, image_path):
        img_np = self._load_img(image_path)
        return self._run_tensorflow_graph(np.expand_dims(img_np, axis=0))

    def infer_batch(self, image_paths):
        img_np = self._load_imgs(image_paths)
        return self._run_tensorflow_graph(img_np)

    def _run_tensorflow_graph(self, image_input):
        ops = self.detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes',
            'detection_scores', 'detection_classes'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.detection_graph.get_tensor_by_name(
                    tensor_name)

        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        output_dict = self.sess.run(tensor_dict,
            feed_dict={image_tensor: image_input})

        # All outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = output_dict['num_detections'].astype(np.int32)
        output_dict['detection_classes'] = output_dict[
            'detection_classes'].astype(np.uint8)

        return output_dict

    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image).reshape(
            (im_height, im_width, model_utils.ModelData.get_input_channels())
        ).astype(np.uint8)

    def _load_imgs(self, image_paths):
        numpy_array = np.zeros((len(image_paths),) + (300, 300, 3))
        for idx, image_path in enumerate(image_paths):
            img_np = self._load_img(image_path)
            numpy_array[idx] = img_np
        return numpy_array

    def _load_img(self, image_path):
        img = Image.open(image_path)
        img_np = self._load_image_into_numpy_array(img)
        return img_np
