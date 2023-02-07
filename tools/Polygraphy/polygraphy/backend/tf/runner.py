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
# Sets up everything needed to perform inference in TensorFlow.
import os
import time
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.backend.base import BaseRunner
from polygraphy.backend.tf import util as tf_util
from polygraphy.logger import G_LOGGER

tf = mod.lazy_import("tensorflow<2.0")


@mod.export()
class TfRunner(BaseRunner):
    """
    Runs inference using a TensorFlow session.
    """

    def __init__(self, sess, timeline_dir=None, name=None):
        """
        Args:
            sess (Union[Tuple[tf.Session, Sequence[str]], Callable() -> Tuple[tf.Session, Sequence[str]]]):
                    A tuple containing a TensorFlow session and output names or a callable that returns one.


            timeline_dir (str):
                    Path to write a TensorFlow timeline.
                    Note that profiling may affect execution time.
            name (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
        """
        super().__init__(name=name, prefix="tf-runner")

        self._sess = sess

        self.timeline_dir = timeline_dir
        self.num_inferences = 0
        self.run_options = None
        self.run_metadata = None
        if self.timeline_dir is not None:
            # Enable profiling
            G_LOGGER.warning("Profiling is enabled. This will impact performance")
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()

    def activate_impl(self):
        (self.sess, self.output_names), _ = util.invoke_if_callable(self._sess)

    def get_input_metadata_impl(self):
        return tf_util.get_input_metadata(self.sess.graph)

    def infer_impl(self, feed_dict):
        G_LOGGER.extra_verbose(f"Received feed_dict: {feed_dict}")
        start = time.time()
        inference_outputs = self.sess.run(
            self.output_names, feed_dict=feed_dict, options=self.run_options, run_metadata=self.run_metadata
        )
        end = time.time()

        out_dict = OrderedDict()
        for name, out in zip(self.output_names, inference_outputs):
            out_dict[name] = out
        self.inference_time = end - start

        if self.timeline_dir is not None:
            from tensorflow.python.client import timeline

            t1 = timeline.Timeline(self.run_metadata.step_stats)

            util.save_file(
                contents=t1.generate_chrome_trace_format(),
                dest=os.path.join(self.timeline_dir, f"run-{self.num_inferences}"),
                mode="w",
            )
        self.num_inferences += 1

        return out_dict

    def deactivate_impl(self):
        self.sess.close()
        del (self.sess, self.output_names)
        self.num_inferences = 0
