#!/usr/bin/env python3
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

import time
import ctypes
import argparse
import numpy as np
import tokenization
import tensorrt as trt
import data_processing as dp
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BERT QA Inference')
    parser.add_argument('-e', '--bert_engine', dest='bert_engine',
            help='Path to BERT TensorRT engine')
    parser.add_argument('-p', '--passage', nargs='*', dest='passage',
            help='Text for paragraph/passage for BERT QA',
            default='')
    parser.add_argument('-pf', '--passage_file', dest='passage_file',
            help='File containing input passage',
            default='')
    parser.add_argument('-q', '--question', nargs='*', dest='question',
            help='Text for query/question for BERT QA',
            default='')
    parser.add_argument('-qf', '--question_file', dest='question_file',
            help='File containiner input question',
            default='')
    parser.add_argument('-v', '--vocab_file', dest='vocab_file',
            help='Path to file containing entire understandable vocab',
            default='./pre-trained_model/uncased_L-24_H-1024_A-16/vocab.txt')
    parser.add_argument('-b', '--batch_size', dest='batch_size',
            help='Batch size for inference', default=1, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if not args.passage == '':
        paragraph_text = ' '.join(args.passage)
    elif not args.passage_file == '':
        f = open(args.passage_file, 'r')
        paragraph_text = f.read()
    else:
        paragraph_text = input("Paragraph: ")

    question_text = None
    if not args.question == '':
        question_text = ' '.join(args.question)
    elif not args.question_file == '':
        f = open(args.question_file, 'r')
        question_text = f.read()

    print("\nPassage: {}".format(paragraph_text))

    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    # The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
    max_query_length = 64
    # When splitting up a long document into chunks, how much stride to take between chunks.
    doc_stride = 128
    # The maximum total input sequence length after WordPiece tokenization.
    # Sequences longer than this will be truncated, and sequences shorter
    max_seq_length = 384
    # Extract tokecs from the paragraph
    doc_tokens = dp.convert_doc_tokens(paragraph_text)

    def question_features(question):
        # Extract features from the paragraph and question
        return dp.convert_examples_to_features(doc_tokens, question, tokenizer, max_seq_length, doc_stride, max_query_length)

    # Import necessary plugins for BERT TensorRT
    nvinfer =  ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
    cm = ctypes.CDLL("/workspace/TensorRT/demo/BERT/build/libcommon.so", mode=ctypes.RTLD_GLOBAL)
    pg = ctypes.CDLL("/workspace/TensorRT/demo/BERT/build/libbert_plugins.so", mode=ctypes.RTLD_GLOBAL)

    with open(args.bert_engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, \
        runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

        def binding_nbytes(binding):
            return trt.volume(engine.get_binding_shape(binding)) * engine.get_binding_dtype(binding).itemsize

        # Allocate device memory for inputs and outputs.
        d_inputs = [cuda.mem_alloc(binding_nbytes(binding)) for binding in engine if engine.binding_is_input(binding)]
        h_output = cuda.pagelocked_empty(tuple(engine.get_binding_shape(3)), dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)

        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        def inference(input_features):
            print("\nRunning Inference...")
            eval_start_time = time.time()

            # Copy inputs
            cuda.memcpy_htod_async(d_inputs[0], input_features["input_ids"], stream)
            cuda.memcpy_htod_async(d_inputs[1], input_features["segment_ids"], stream)
            cuda.memcpy_htod_async(d_inputs[2], input_features["input_mask"], stream)

            # Run inference
            context.execute_async(bindings=[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
            # Transfer predictions back from GPU
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            # Synchronize the stream
            stream.synchronize()

            eval_time_elapsed = time.time() - eval_start_time

            # Data Post-processing
            start_logits = h_output[:, 0]
            end_logits = h_output[:, 1]

            # Total number of n-best predictions to generate in the nbest_predictions.json output file
            n_best_size = 20

            # The maximum length of an answer that can be generated. This is needed
            # because the start and end predictions are not conditioned on one another
            max_answer_length = 30

            prediction, nbest_json, scores_diff_json = dp.get_predictions(doc_tokens, input_features,
                    start_logits, end_logits, n_best_size, max_answer_length)

            print("------------------------")
            print("Running inference in {:.3f} Sentences/Sec".format(1.0/eval_time_elapsed))
            print("------------------------")

            print("Answer: '{}'".format(prediction))
            print("With probability: {:.3f}".format(nbest_json[0]['probability']*100.0))

        if question_text:
            print("\nQuestion: {}".format(question_text))
            features = question_features(question_text)
            inference(features)
        else:
            # If no question text is provided, loop until the question is 'exit'
            EXIT_CMDS = ["exit", "quit"]
            question_text = input("Question (to exit, type one of {:}): ".format(EXIT_CMDS))
            while question_text.strip() not in EXIT_CMDS:
                features = question_features(question_text)
                inference(features)
                question_text = input("Question (to exit, type one of {:}): ".format(EXIT_CMDS))
