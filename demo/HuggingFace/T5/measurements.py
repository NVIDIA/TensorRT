"""
Utils specific to T5 network.
"""

# torch
import torch

# numpy
import numpy as np

# numpy
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)

# TRT-HuggingFace
from NNDF.general_utils import measure_python_inference_code
from NNDF.torch_utils import use_cuda


@use_cuda
def decoder_inference(
    t5_decoder, input_ids, encoder_last_hidden_state, timing_profile, use_cuda=True
):
    decoder_stmt = lambda: t5_decoder(
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
    use_cuda=True,
):
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length)])
    decoder_input_ids = torch.full(
        (1, 1), tokenizer.convert_tokens_to_ids(tokenizer.pad_token), dtype=torch.int32
    )

    if use_cuda:
        decoder_input_ids = decoder_input_ids.to("cuda")

    def _e2e():
        encoder_last_hidden_state = t5_encoder(input_ids=input_ids)

        return t5_decoder.greedy_search(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_last_hidden_state,
            stopping_criteria=stopping_criteria,
        )

    full_e2e_median_time = measure_python_inference_code(
        _e2e,
        number=timing_profile.number,
        iterations=timing_profile.iterations,
    )

    return (_e2e(), full_e2e_median_time)
