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
Utils specific to T5 network.
"""

# torch
import torch

# from HuggingFace transformers
from transformers.generation_logits_process import (
    MinLengthLogitsProcessor,
    LogitsProcessorList,
    ForcedEOSTokenLogitsProcessor,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from transformers.generation_beam_search import (
    BeamSearchScorer,
)

# TRT-HuggingFace
from NNDF.general_utils import measure_python_inference_code
from NNDF.torch_utils import use_cuda, expand_inputs_for_beam_search
from NNDF.tensorrt_utils import TRTNativeRunner
from NNDF.logger import G_LOGGER

@use_cuda
def decoder_inference(
    t5_decoder, input_ids, encoder_last_hidden_state, timing_profile, use_cuda=True, use_cache=False, past_key_values=None
):
    # This implementation is a bit ugly. Moving implementation of the model to check HFRunner would be cleaner.
    if isinstance(t5_decoder, TRTNativeRunner):
        # Function is technically in T5TRTDecoder however due to circular import, TRTNativeRunner in this module scope
        # implies the existence of this function.
        t5_decoder.set_encoder_hidden_states_for_inference_cycle(encoder_last_hidden_state)
        t5_decoder.set_return_device("cuda" if use_cuda else "cpu")

    def decoder_stmt():
        t5_decoder(
            input_ids=input_ids, encoder_hidden_states=encoder_last_hidden_state, use_cache=use_cache, 
            past_key_values=past_key_values
        )

    decoder_e2e_time = measure_python_inference_code(decoder_stmt, timing_profile)

    return (decoder_stmt(), decoder_e2e_time)


@use_cuda
def encoder_inference(t5_encoder, input_ids, timing_profile, use_cuda=True):
    encoder_stmt = lambda: t5_encoder(input_ids=input_ids)
    encoder_e2e_time = measure_python_inference_code(encoder_stmt, timing_profile)

    return (encoder_stmt(), encoder_e2e_time)


# Code specifically for Pythonic inference measurement used across all T5 related scripts
@use_cuda
def full_inference_greedy(
    t5_encoder,
    t5_decoder,
    input_ids,
    tokenizer,
    timing_profile,
    max_length,
    min_length=0,
    batch_size=1,
    use_cuda=True,
    early_stopping=False,
    use_cache=False
):
    G_LOGGER.info("Running full inference with greedy decoding...")

    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length)])

    logits_processor = LogitsProcessorList([
        MinLengthLogitsProcessor(min_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token)),
        ForcedEOSTokenLogitsProcessor(max_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
    ])
    
    decoder_input_ids = torch.full(
        (batch_size, 1), tokenizer.convert_tokens_to_ids(tokenizer.pad_token), dtype=torch.int32
    )

    if use_cuda:
        decoder_input_ids = decoder_input_ids.to("cuda")
    else:
        decoder_input_ids = decoder_input_ids.to("cpu")

    def _e2e():
        with torch.no_grad():
            encoder_last_hidden_state = t5_encoder(input_ids=input_ids)
            decoder_output_greedy = t5_decoder.greedy_search(
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
            encoder_last_hidden_state = t5_encoder(input_ids=input_ids)
            t5_decoder.set_encoder_hidden_states_for_inference_cycle(encoder_last_hidden_state)
            decoder_output_greedy = t5_decoder.greedy_search(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_last_hidden_state,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=use_cache
            )
        return decoder_output_greedy
        
    measurement_function = _e2e
    if isinstance(t5_decoder, TRTNativeRunner):
        t5_decoder.set_return_device("cuda" if use_cuda else "cpu")
        measurement_function = _e2e_trt

    full_e2e_time = measure_python_inference_code(measurement_function, timing_profile)

    return (measurement_function(), full_e2e_time)

@use_cuda
def full_inference_beam(
    t5_encoder,
    t5_decoder,
    input_ids,
    tokenizer,
    timing_profile,
    num_beams,
    max_length,
    min_length=0,
    batch_size=1,
    use_cuda=True,
    early_stopping=False,
    use_cache=False
):

    G_LOGGER.info(f"Running full inference with beam search (num_beams = {num_beams}) decoding...")

    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length)])

    logits_processor = LogitsProcessorList([
        MinLengthLogitsProcessor(min_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token)),
        ForcedEOSTokenLogitsProcessor(max_length, tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
    ])
    
    decoder_input_ids = torch.full(
        (batch_size, 1), tokenizer.convert_tokens_to_ids(tokenizer.pad_token), dtype=torch.int32
    )

    decoder_input_ids = expand_inputs_for_beam_search(decoder_input_ids, expand_size=num_beams)

    if use_cuda:
        decoder_input_ids = decoder_input_ids.to("cuda")
    else:
        decoder_input_ids = decoder_input_ids.to("cpu")

    def _e2e():
        with torch.no_grad():
            # beam scorer must be reset before each beam search run, otherwise beam search will be skipped due to scorer cache
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device="cuda" if use_cuda else "cpu",
                do_early_stopping=early_stopping,
            )

            encoder_last_hidden_state = t5_encoder(input_ids=input_ids)
            
            encoder_last_hidden_state = expand_inputs_for_beam_search(encoder_last_hidden_state, expand_size=num_beams)

            decoder_output_beam = t5_decoder.beam_search(
                input_ids=decoder_input_ids,
                beam_scorer=beam_scorer,
                encoder_hidden_states=encoder_last_hidden_state,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=use_cache
            )
        return decoder_output_beam

    # With e2e we can opt to bind inputs only once for hidden states for optimization
    def _e2e_trt():
        with torch.no_grad():
            # beam scorer must be reset before each beam search run, otherwise beam search will be skipped due to scorer cache
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device="cuda" if use_cuda else "cpu",
                do_early_stopping=early_stopping
            )

            encoder_last_hidden_state = t5_encoder(input_ids=input_ids)
            
            encoder_last_hidden_state = expand_inputs_for_beam_search(encoder_last_hidden_state, expand_size=num_beams)
            
            t5_decoder.set_encoder_hidden_states_for_inference_cycle(encoder_last_hidden_state)
            decoder_output_beam = t5_decoder.beam_search(
                input_ids=decoder_input_ids,
                beam_scorer=beam_scorer,
                encoder_hidden_states=encoder_last_hidden_state,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                use_cache=use_cache
            )
        return decoder_output_beam

    measurement_function = _e2e
    if isinstance(t5_decoder, TRTNativeRunner):
        t5_decoder.set_return_device("cuda" if use_cuda else "cpu")
        measurement_function = _e2e_trt

    full_e2e_time = measure_python_inference_code(measurement_function, timing_profile)

    return (measurement_function(), full_e2e_time)

@use_cuda
def calculate_perplexity(
    t5_encoder,
    t5_decoder,
    tokenizer,
    input_ids,
    decoder_input_ids,
    max_seq_len=None,
    use_cuda=True,
):
    encoder_last_hidden_state = t5_encoder(input_ids=input_ids)
    if isinstance(t5_decoder, TRTNativeRunner):
        t5_decoder.set_return_device("cuda" if use_cuda else "cpu")
        t5_decoder.set_encoder_hidden_states_for_inference_cycle(encoder_last_hidden_state)

    # Set the first token to be pad token
    decoder_input_ids_padded = torch.full(
        decoder_input_ids.size()[:-1] + (decoder_input_ids.size()[-1] + 1,),
        tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
        dtype=decoder_input_ids.dtype,
    )
    decoder_input_ids_padded[..., 1:] = decoder_input_ids

    if use_cuda:
        encoder_last_hidden_state = encoder_last_hidden_state.to("cuda")
        decoder_input_ids_padded = decoder_input_ids_padded.to("cuda")

    with torch.no_grad():
        if max_seq_len is not None:
            decoder_input_ids_padded = decoder_input_ids_padded[:, :max_seq_len]
        logits = t5_decoder(decoder_input_ids_padded, encoder_last_hidden_state, return_dict=True).logits
        # Truncate the last prediction
        logits = logits[:, :-1, :]
        loss = torch.nn.CrossEntropyLoss()(logits.permute((0, 2, 1)), decoder_input_ids)
        return torch.exp(loss).item()
