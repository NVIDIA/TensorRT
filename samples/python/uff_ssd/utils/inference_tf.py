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

# Utility functions for performing image inference

import tensorflow as tf
from PIL import Image
import numpy as np

import utils.model as model_utils  # UFF conversion uttils

# This class is similar as TRTInference inference, but it manages Tensorflow
class TensorflowInference(object):
    def __init__(self, pb_model_path):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(pb_model_path, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
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
        for key in ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]:
            tensor_name = key + ":0"
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.detection_graph.get_tensor_by_name(tensor_name)

        image_tensor = self.detection_graph.get_tensor_by_name("image_tensor:0")
        output_dict = self.sess.run(tensor_dict, feed_dict={image_tensor: image_input})

        # All outputs are float32 numpy arrays, so convert types as appropriate
        output_dict["num_detections"] = output_dict["num_detections"].astype(np.int32)
        output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.uint8)

        return output_dict

    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return (
            np.array(image).reshape((im_height, im_width, model_utils.ModelData.get_input_channels())).astype(np.uint8)
        )

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
