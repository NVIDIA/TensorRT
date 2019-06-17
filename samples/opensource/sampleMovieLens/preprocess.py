#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
import graphsurgeon as gs
import tensorflow as tf

def preprocess(dynamic_graph):
    axis = dynamic_graph.find_nodes_by_path("concatenate/concat/axis")[0]
    # Set axis to 2, because of discrepancies between TensorFlow and TensorRT.
    axis.attr["value"].tensor.int_val[0] = 2
