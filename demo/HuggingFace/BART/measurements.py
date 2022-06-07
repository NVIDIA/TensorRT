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
Utils specific to BART network.
"""

# torch
import torch

# from HuggingFace transformers
from transformers.generation_logits_process import (
    NoRepeatNGramLogitsProcessor,
    MinLengthLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    LogitsProcessorList,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)

from BART.BARTModelConfig import BARTModelTRTConfig

# TRT-HuggingFace
from NNDF.general_utils import measure_python_inference_code
from NNDF.torch_utils import use_cuda
from NNDF.tensorrt_utils import TRTNativeRunner


@use_cuda
def decoder_inference(
    BART_decoder, input_ids, encoder_last_hidden_state, timing_profile, use_cuda=True, use_cache=True
):
    # This implementation is a bit ugly. Moving implementation of the model to check HFRunner would be cleaner.
    if isinstance(BART_decoder, TRTNativeRunner):
        # Function is technically in BARTTRTDecoder however due to circular import, TRTNativeRunner in this module scope
        # implies the existence of this function.
        BART_decoder.set_encoder_hidden_states_for_inference_cycle(encoder_last_hidden_state)
        BART_decoder.set_return_device("cuda" if use_cuda else "cpu")

    def decoder_stmt():
        BART_decoder(
            input_ids=input_ids, encoder_hidden_states=encoder_last_hidden_state, use_cache=use_cache
        )

    decoder_e2e_median_time = measure_python_inference_code(decoder_stmt, timing_profile)

    return (decoder_stmt(), decoder_e2e_median_time)


@use_cuda
def encoder_inference(BART_encoder, input_ids, timing_profile, use_cuda=True):
    encoder_stmt = lambda: BART_encoder(input_ids=input_ids)
    encoder_e2e_median_time = measure_python_inference_code(encoder_stmt, timing_profile)

    return (encoder_stmt(), encoder_e2e_median_time)


# Code specifically for Pythonic inference measurement used across all BART related scripts
@use_cuda
def full_inference_greedy(
    BART_encoder,
    BART_decoder,
    input_ids,
    tokenizer,
    timing_profile,
    max_length,
    batch_size=1,
    use_cuda=True,
    early_stopping=True,
    use_cache=True
):
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length)])
    no_repeat_ngram_size = BARTModelTRTConfig.NO_REPEAT_NGRAM_SIZE
    min_length = BARTModelTRTConfig.MIN_OUTPUT_LENGTH[tokenizer.name_or_path] # instead of using metadata.variant (which require passing metadata), I just hacked here to get the variant name from tokenizer
    logits_processor = LogitsProcessorList([
        NoRepeatNGramLogitsProcessor(no_repeat_ngram_size), 
        MinLengthLogitsProcessor(min_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token)),
        ForcedBOSTokenLogitsProcessor(tokenizer.convert_tokens_to_ids(tokenizer.bos_token)),
        ForcedEOSTokenLogitsProcessor(max_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
    ]) # by checking HuggingFace's generate() implementation carefully, the default logits processor for BART has no_repeat_ngram_size = 3 and forced_eos_token_id = 2. In this way we can get identical results with raw HuggingFace

    decoder_input_ids = torch.full(
        (batch_size, 1), tokenizer.convert_tokens_to_ids(tokenizer.eos_token), dtype=torch.int32
    )

    if use_cuda:
        decoder_input_ids = decoder_input_ids.to("cuda")
    else:
        decoder_input_ids = decoder_input_ids.to("cpu")

    def _e2e():
        with torch.no_grad():
            encoder_last_hidden_state = BART_encoder(input_ids=input_ids)
            decoder_output_greedy = BART_decoder.greedy_search(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_last_hidden_state,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=use_cache
            )
        return decoder_output_greedy

    # With e2e we can opt to bind inputs only once for hidden states for optimization
    def _e2e_trt():
        with torch.no_grad():
            encoder_last_hidden_state = BART_encoder(input_ids=input_ids)
            BART_decoder.set_encoder_hidden_states_for_inference_cycle(encoder_last_hidden_state)
            decoder_output_greedy = BART_decoder.greedy_search(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_last_hidden_state,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=use_cache
            )
        return decoder_output_greedy

    measurement_function = _e2e
    if isinstance(BART_decoder, TRTNativeRunner):
        BART_decoder.set_return_device("cuda" if use_cuda else "cpu")
        measurement_function = _e2e_trt

    full_e2e_median_time = measure_python_inference_code(measurement_function, timing_profile)

    return (measurement_function(), full_e2e_median_time)
