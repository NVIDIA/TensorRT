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

# TRT-HuggingFace
from NNDF.general_utils import measure_python_inference_code
from NNDF.torch_utils import use_cuda, expand_inputs_for_beam_search
from NNDF.tensorrt_utils import TRTNativeRunner
from NNDF.logger import G_LOGGER
from transformers.modeling_outputs import BaseModelOutput

@use_cuda
def decoder_inference(
    t5_decoder, input_ids, encoder_last_hidden_state, timing_profile, use_cuda=True, use_cache=False, past_key_values=None
):
    # This implementation is a bit ugly. Moving implementation of the model to check HFRunner would be cleaner.
    if isinstance(t5_decoder, TRTNativeRunner):
        # Function is technically in T5TRTDecoder however due to circular import, TRTNativeRunner in this module scope
        # implies the existence of this function.
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

@use_cuda
def full_inference(
    t5_encoder,
    t5_decoder,
    input_ids,
    tokenizer,
    timing_profile,
    max_length,
    min_length=0,
    num_beams=1,
    batch_size=1,
    use_cuda=True,
    early_stopping=True,
    use_cache=False
):

    G_LOGGER.info(f"Running full inference...")
    encoder_last_hidden_state = t5_encoder(input_ids=input_ids)

    def _e2e():
        with torch.no_grad():
            decoder_output = t5_decoder.generate(
                input_ids,
                max_length = max_length,
                min_length = min_length,
                num_beams = num_beams,
                early_stopping = early_stopping,
                eos_token_id = t5_decoder.config.eos_token_id,
                pad_token_id = t5_decoder.config.pad_token_id,
                use_cache = use_cache,
                encoder_outputs = BaseModelOutput(last_hidden_state = encoder_last_hidden_state),
            )
        return decoder_output

    if isinstance(t5_decoder, TRTNativeRunner):
        t5_decoder.set_return_device("cuda" if use_cuda else "cpu")

    measurement_function = _e2e

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
