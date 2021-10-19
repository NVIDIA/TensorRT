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

"""
Utils specific to T5 network.
"""

# torch
import torch

# numpy
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)

# TRT-HuggingFace
from NNDF.general_utils import measure_python_inference_code
from NNDF.torch_utils import use_cuda
from NNDF.tensorrt_utils import TRTNativeRunner


@use_cuda
def decoder_inference(
    t5_decoder, input_ids, encoder_last_hidden_state, timing_profile, use_cuda=True
):
    # This implementation is a bit ugly. Moving implementation of the model to check HFRunner would be cleaner.
    if isinstance(t5_decoder, TRTNativeRunner):
        # Function is technically in T5TRTDecoder however due to circular import, TRTNativeRunner in this module scope
        # implies the existence of this function.
        t5_decoder.set_encoder_hidden_states_for_inference_cycle(encoder_last_hidden_state)
        t5_decoder.set_return_device("cuda" if use_cuda else "cpu")

    def decoder_stmt():
        t5_decoder(
            input_ids=input_ids, encoder_hidden_states=encoder_last_hidden_state
        )

    decoder_e2e_median_time = measure_python_inference_code(
        decoder_stmt, number=timing_profile.number, iterations=timing_profile.iterations
    )

    return (decoder_stmt(), decoder_e2e_median_time)


@use_cuda
def encoder_inference(t5_encoder, input_ids, timing_profile, use_cuda=True):
    encoder_stmt = lambda: t5_encoder(input_ids=input_ids)
    encoder_e2e_median_time = measure_python_inference_code(
        encoder_stmt, number=timing_profile.number, iterations=timing_profile.iterations
    )

    return (encoder_stmt(), encoder_e2e_median_time)


# Code specifically for Pythonic inference measurement used across all T5 related scripts
@use_cuda
def full_inference_greedy(
    t5_encoder,
    t5_decoder,
    input_ids,
    tokenizer,
    timing_profile,
    max_length,
    batch_size=1,
    use_cuda=True,
):
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length)])
    decoder_input_ids = torch.full(
        (batch_size, 1), tokenizer.convert_tokens_to_ids(tokenizer.pad_token), dtype=torch.int32
    )

    if use_cuda:
        decoder_input_ids = decoder_input_ids.to("cuda")
    else:
        decoder_input_ids = decoder_input_ids.to("cpu")

    def _e2e():
        encoder_last_hidden_state = t5_encoder(input_ids=input_ids)
        return t5_decoder.greedy_search(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_last_hidden_state,
            stopping_criteria=stopping_criteria,
        )

    # With e2e we can opt to bind inputs only once for hidden states for optimization
    def _e2e_trt():
        encoder_last_hidden_state = t5_encoder(input_ids=input_ids)
        t5_decoder.set_encoder_hidden_states_for_inference_cycle(encoder_last_hidden_state)
        return t5_decoder.greedy_search(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_last_hidden_state,
            stopping_criteria=stopping_criteria,
        )

    measurement_function = _e2e
    if isinstance(t5_decoder, TRTNativeRunner):
        t5_decoder.set_return_device("cuda" if use_cuda else "cpu")
        measurement_function = _e2e_trt

    full_e2e_median_time = measure_python_inference_code(
        measurement_function,
        number=timing_profile.number,
        iterations=timing_profile.iterations,
    )

    return (measurement_function(), full_e2e_median_time)
