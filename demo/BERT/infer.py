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

import argparse
import ctypes
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np

import helpers.tokenization as tokenization
import helpers.data_processing as dp
import pdb

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

class DeviceBuffer(object):
    def __init__(self, shape, dtype=trt.int32):
        self.buf = cuda.mem_alloc(trt.volume(shape) * dtype.itemsize)

    def binding(self):
        return int(self.buf)

    def free(self):
        self.buf.free()


def main():
    parser = argparse.ArgumentParser(description='BERT Inference Benchmark')
    parser.add_argument("-e", "--engine", help='Path to BERT TensorRT engine')
    parser.add_argument('-b', '--batch-size', default=[], action="append", help='Batch size(s) to benchmark. Can be specified multiple times for more than one batch size. This script assumes that the engine has been built with one optimization profile for each batch size, and that these profiles are in order of increasing batch size.', type=int)
    parser.add_argument('-s', '--sequence-length', default=128, help='Sequence length of the BERT model', type=int)
    parser.add_argument('-i', '--iterations', default=1, help='Number of iterations to run when benchmarking each batch size.', type=int)
    parser.add_argument('-w', '--warm-up-runs', default=0, help='Number of iterations to run prior to benchmarking.', type=int)
    parser.add_argument('-r', '--random-seed', required=False, default=12345, help='Random seed.', type=int)
    parser.add_argument('-p', '--passage', nargs='*', help='Text for paragraph/passage for BERT QA', default='')
    parser.add_argument('-q', '--question', nargs='*', help='Text for query/question for BERT QA', default='')
    parser.add_argument('-v', '--vocab-file', help='Path to file containing entire understandable vocab')

    args, _ = parser.parse_known_args()
    args.batch_size = args.batch_size or [1]

    # Import necessary plugins for BERT TensorRT
    ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)

    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
        # Allocate buffers large enough to store the largest batch size
        max_input_shape = (args.sequence_length, max(args.batch_size))
        max_output_shape = (args.sequence_length, max(args.batch_size), 2, 1, 1)
        buffers = [
            DeviceBuffer(max_input_shape),
            DeviceBuffer(max_input_shape),
            DeviceBuffer(max_input_shape),
            DeviceBuffer(max_output_shape)
        ]

        def question_features(tokens, question):
            # Extract features from the paragraph and question
            tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
            return dp.convert_example_to_features(tokens, question, tokenizer, args.sequence_length, 128, 64)

        # Prepare random input
        pseudo_vocab_size = 30522
        pseudo_type_vocab_size = 2
        np.random.seed(args.random_seed)

        paragraph_text = ' '.join(args.passage)
        question_text = ' '.join(args.question)
        print("\nPassage: {}".format(paragraph_text))
        print("\nQuestion: {}".format(question_text))
        doc_tokens = dp.convert_doc_tokens(paragraph_text)
        features = question_features(doc_tokens, question_text)
        test_word_ids = features[0].input_ids
        test_segment_ids = features[0].segment_ids
        test_input_mask = features[0].input_mask

        # Copy input h2d
        cuda.memcpy_htod(buffers[0].buf, test_word_ids.ravel())
        cuda.memcpy_htod(buffers[1].buf, test_segment_ids.ravel())
        cuda.memcpy_htod(buffers[2].buf, test_input_mask.ravel())

        num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles

        bench_times = {}

        for idx, batch_size in enumerate(sorted(args.batch_size)):
            context.active_optimization_profile = idx

            # Each profile has unique bindings
            binding_idx_offset = idx * num_binding_per_profile
            bindings = [0] * binding_idx_offset + [buf.binding() for buf in buffers]

            shapes = {
                "input_ids": (args.sequence_length, batch_size),
                "segment_ids": (args.sequence_length, batch_size),
                "input_mask": (args.sequence_length, batch_size),
            }

            for binding, shape in shapes.items():
                context.set_binding_shape(engine[binding] + binding_idx_offset, shape)
            assert context.all_binding_shapes_specified

            # Inference
            total_time = 0
            start = cuda.Event()
            end = cuda.Event()
            stream = cuda.Stream()

            # Warmup
            for _ in range(args.warm_up_runs):
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                stream.synchronize()

            # Timing loop
            times = []
            for _ in range(args.iterations):
                start.record(stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                end.record(stream)
                stream.synchronize()
                times.append(end.time_since(start))

            # Transfer predictions back from GPU
            cuda.memcpy_dtoh_async(h_output, d_output, stream)

                for index, batch in enumerate(h_output):
                    # Data Post-processing
                    networkOutputs.append(_NetworkOutput(
                        start_logits = np.array(batch.squeeze()[:, 0]),
                        end_logits = np.array(batch.squeeze()[:, 1]),
                        feature_index = feature_index
                        ))

            # Compute average time, 95th percentile time and 99th percentile time.
            bench_times[batch_size] = times

        [b.free() for b in buffers]

        for batch_size, times in bench_times.items():
            total_time = sum(times)
            avg_time = total_time / float(len(times))
            times.sort()
            percentile95 = times[int(len(times) * 0.95)]
            percentile99 = times[int(len(times) * 0.99)]
            print("Running {:} iterations with Batch Size: {:}\n\tTotal Time: {:} ms \tAverage Time: {:} ms\t95th Percentile Time: {:} ms\t99th Percentile Time: {:}".format(args.iterations, batch_size, total_time, avg_time, percentile95, percentile99))


if __name__ == '__main__':
    main()
