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

# uff_ssd path management singleton class
import os
import sys
import tensorrt as trt


class Paths(object):
    def __init__(self):
        self._SAMPLE_ROOT = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir
        )
        self._WORKSPACE_DIR_PATH = os.path.join(
            self._SAMPLE_ROOT,
            'workspace'
        )
        self._VOC_DIR_PATH = \
            os.path.join(self._SAMPLE_ROOT, 'VOCdevkit', 'VOC2007')

    # User configurable paths

    def set_workspace_dir_path(self, workspace_dir):
        self._WORKSPACE_DIR_PATH = workspace_dir

    def get_workspace_dir_path(self):
        return self._WORKSPACE_DIR_PATH

    def set_voc_dir_path(self, voc_dir_path):
        self._VOC_DIR_PATH = voc_dir_path

    def get_voc_dir_path(self):
        return self._VOC_DIR_PATH

    # Fixed paths

    def get_sample_root(self):
        return self._SAMPLE_ROOT

    def get_models_dir_path(self):
        return os.path.join(self.get_workspace_dir_path(), 'models')

    def get_engines_dir_path(self):
        return os.path.join(self.get_workspace_dir_path(), 'engines')

    def get_engine_path(self, inference_type=trt.DataType.FLOAT, max_batch_size=1):
        inference_type_to_str = {
            trt.DataType.FLOAT: 'FLOAT',
            trt.DataType.HALF: 'HALF',
            trt.DataType.INT32: 'INT32',
            trt.DataType.INT8: 'INT8'
        }
        return os.path.join(
            self.get_engines_dir_path(),
            inference_type_to_str[inference_type],
            'engine_bs_{}.buf'.format(max_batch_size))

    def get_voc_annotation_cache_path(self):
        return os.path.join(self.get_workspace_dir_path(), 'annotations_cache')

    def get_voc_image_set_path(self):
        return os.path.join(self.get_voc_dir_path(), 'ImageSets', 'Main', 'test.txt')

    def get_voc_annotation_path(self):
        return os.path.join(self.get_voc_dir_path(), 'Annotations', '{}.xml')

    def get_voc_ppm_img_path(self):
        return os.path.join(self.get_voc_dir_path(), 'PPMImages', '{}.ppm')

    def get_voc_jpg_img_path(self):
        return os.path.join(self.get_voc_dir_path(), 'JPEGImages', '{}.jpg')

    def get_voc_tensorflow_model_detections_path(self):
        return os.path.join(self.get_workspace_dir_path(), 'results', 'tensorflow')

    def get_voc_tensorrt_model_detections_path(self, trt_engine_datatype=trt.DataType.FLOAT):
        trt_results_path = \
            os.path.join(self.get_workspace_dir_path(), 'results', 'tensorrt')
        if trt_engine_datatype == trt.DataType.HALF:
            return os.path.join(trt_results_path, 'HALF')
        else:
            return os.path.join(trt_results_path, 'FLOAT')

    def get_voc_model_detections_path(self, backend='tensorrt', use_fp16=False):
        if backend != 'tensorrt':
            return self.get_voc_tensorflow_model_detections_path()
        else:
            return self.get_voc_tensorrt_model_detections_path(use_fp16)

    def get_model_url(self, model_name):
        return 'http://download.tensorflow.org/models/object_detection/{}.tar.gz'.format(model_name)

    def get_model_dir_path(self, model_name):
        return os.path.join(self.get_models_dir_path(), model_name)

    def get_model_pb_path(self, model_name):
        return os.path.join(
            self.get_model_dir_path(model_name),
            'frozen_inference_graph.pb'
        )

    def get_model_uff_path(self, model_name):
        return os.path.join(
            self.get_model_dir_path(model_name),
            'frozen_inference_graph.uff'
        )

    # Paths correctness verifier

    def verify_all_paths(self, should_verify_voc=False):
        error = False

        if should_verify_voc:
            error = self._verify_voc_paths()
        if not os.path.exists(self.get_workspace_dir_path()):
            error = True

        if error:
            print("An error occured when running the script.")
            sys.exit(1)

    def _verify_voc_paths(self):
        error = False
        voc_dir = self.get_voc_dir_path()
        voc_image_list = self.get_voc_image_set_path()
        # 1) Check if directory and image list file are present
        if not os.path.exists(voc_dir) or \
            not os.path.exists(voc_image_list):
            self._print_incorrect_voc_error(voc_dir)
            error = True
        # 2) Check if all images listed in image list are present
        with open(voc_image_list, 'r') as f:
            image_numbers = f.readlines()
            image_numbers = [line.strip() for line in image_numbers]
        if not self._verify_voc(image_numbers):
            self._print_incorrect_voc_error(voc_dir)
            error = True
        return error

    def _verify_voc(self, voc_image_list):
        voc_image_path = self.get_voc_jpg_img_path()
        for img_number in voc_image_list:
            img = voc_image_path.format(img_number)
            if not os.path.exists(img):
                return False
            return True


    # Error printers

    def _print_incorrect_voc_error(self, voc_dir):
        print(
            "Error: {}\n{}\n{}".format(
                "Incomplete VOC dataset detected (voc_dir: {})".format(voc_dir),
                "Try redownloading VOC or check if --voc_dir is set up correctly",
                "For more details, check README.md"
            )
        )


PATHS = Paths()
