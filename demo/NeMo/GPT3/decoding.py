#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import Iterable
import sys
from typing import List

from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator
from megatron.core import parallel_state
from nemo.collections.nlp.modules.common.text_generation_strategy import GPTModelTextGenerationStrategy
from nemo.utils import AppState
import torch
import torch.nn.functional as F

from GPT3.trt_utils import GPTTRTDecoder

sys.path.append('../../HuggingFace') # Include HuggingFace
from NNDF.logger import G_LOGGER


def sample_sequence_batch(
    model,
    inference_strategy,
    context_tokens,
    context_lengths,
    tokens_to_generate,
    all_probs=False,
    temperature=None,
    extra={},
):
    def repetition_penalty(logits, repetition_penalty, used_tokens):
        """ Implement the repetition penalty, check paper
        https://arxiv.org/pdf/1909.05858.pdf
        """
        if used_tokens is not None and repetition_penalty != 1.0:
            logits_update = torch.gather(logits, 1, used_tokens)
            logits = torch.scatter(logits, 1, used_tokens, logits_update / repetition_penalty)
        return logits

    def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf'), started=None):
        """
        This function has been mostly taken from huggingface conversational
            ai code at
            https://medium.com/huggingface/how-to-build-a-state-of-the-art-
                conversational-ai-with-transfer-learning-2d818ac26313

            @param logits: logits tensor
            @param top_k: keep only top k tokens with highest probability
            @param top_p: keep the top tokens with cumulative probability
            @filter_value: value to set filtered tokens to
            @started: a tensor of bools indicating whether the text generation starts for the batch
            returns the filtered logits
        """
        if top_k > 0:
            # Remove all tokens with a probability less than the
            # last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            if started is not None:
                for i in torch.arange(indices_to_remove.size(0))[started]:
                    logits[i, indices_to_remove[i]] = filter_value
            else:
                logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            # Cconvert to 1D
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token
            # above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            if started is not None:
                for i in torch.arange(sorted_indices.size(0))[started]:
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = filter_value
            else:
                for i in range(sorted_indices.size(0)):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = filter_value

        return logits

    app_state = AppState()
    batch_size = context_tokens.shape[0]
    if not (hasattr(model, "trt") or hasattr(model, "onnx")):
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=batch_size,
            micro_batch_size=batch_size,
            data_parallel_size=1,
        )

    tokenizer = model.tokenizer
    # initialize the batch
    with torch.no_grad():
        context_length = context_lengths.min().item()
        context_lengths_cpu = context_lengths.cpu()
        inference_strategy.init_batch(context_tokens, context_length)
        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eod_id = tokenizer.eos_id
        counter = 0

        tokens = context_tokens
        output_logits = None
        all_generated_indices = None  # used to track all generated indices
        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + context_lengths.max().item()
        maxlen = inference_strategy.clip_max_len(maxlen)

        is_done = torch.zeros([batch_size]).byte()
        lengths = torch.ones([batch_size]).long() * maxlen

        use_cache = extra.get("use_cache", False)
        is_onnx = hasattr(model, "onnx")
        is_trt = hasattr(model, "trt")

        if is_trt:
            assert isinstance(model.trt, GPTTRTDecoder)
            input_ids_name = model.trt.get_input_ids_name()
            input_ids_type = model.trt.get_torch_type(input_ids_name)
            position_ids_name = model.trt.get_position_ids_name()
            position_ids_type =  model.trt.get_torch_type(position_ids_name)
            attention_mask_name = model.trt.get_attention_mask_name()
            if attention_mask_name != None:
                attention_mask_type = model.trt.get_torch_type(attention_mask_name)

            position_ids = inference_strategy.position_ids
            attention_mask = inference_strategy.attention_mask

        torch.cuda.nvtx.range_pop() # "Prepare Batch"
        while context_length < maxlen:
            torch.cuda.nvtx.range_push("I/O Setup")

            output = None
            if is_onnx and use_cache:
                G_LOGGER.warn(f"ONNX runtime path does not support KV-cache.")

            # Modify counter based on using cache or not.
            if is_trt:
                # TRT input preprocessing doesn't use nemo function
                pass
            elif not is_onnx and use_cache:
                batch, tensor_shape = inference_strategy.prepare_batch_at_step(
                    tokens, maxlen, batch_size, counter, context_length
                )
            else:
                batch, tensor_shape = inference_strategy.prepare_batch_at_step(
                    tokens, maxlen, batch_size, 0, context_length # step is always 0
                )

            # inputs input_ids: [BS, SEQ], position_ids: [BS, SEQ], attention_mask: [1, 1, SEQ, SEQ]
            if is_trt:
                context_mode = (use_cache and counter == 0) or not use_cache
                if context_mode or not use_cache:
                    # context mode
                    batch_tokens = tokens[:, :context_length]
                    batch_position_ids = position_ids[:, :context_length]
                else:
                    # generate mode
                    batch_tokens = tokens[:, context_length - 1].view(batch_size, -1)
                    batch_position_ids = position_ids[:, context_length - 1].view(batch_size, -1)
                seq_len = batch_tokens.shape[1]
                batch_attention_mask = attention_mask[0:1, 0:1, :seq_len, :seq_len]
                input_ids = batch_tokens.type(input_ids_type).contiguous().cuda()
                tensor_dict = {input_ids_name : (input_ids.data_ptr(), input_ids.shape)}
                if position_ids_name != None:
                    batch_position_ids = batch_position_ids.type(position_ids_type).contiguous().cuda()
                    tensor_dict[position_ids_name] = (batch_position_ids.data_ptr(), batch_position_ids.shape)
                if attention_mask_name != None:
                    batch_attention_mask = batch_attention_mask.type(attention_mask_type).contiguous().cuda()
                    tensor_dict[attention_mask_name] = (batch_attention_mask.data_ptr(), batch_attention_mask.shape)

                logits_name = model.trt.get_output_name()
                torch.cuda.nvtx.range_pop() # "I/O Setup"
                output = model.trt.run(logits_name, tensor_dict, seq_len, context_mode)

            elif is_onnx:
                assert len(batch) == 5, "Length of batch must be 5."
                (
                    batch_tokens,
                    attention_mask,
                    position_ids,
                    set_inference_key_value_memory,
                    _,
                ) = batch
                seq_len = batch_tokens.shape[1]
                attention_mask = attention_mask[0:1, 0:1, 0:seq_len, 0:seq_len]

                from onnxruntime import InferenceSession
                assert isinstance(model.onnxrt, InferenceSession)
                # Currently only support onnx runtime with cpu
                # Our fp8 models don't currently use a user-provided attention_mask
                tensor_dict = {'input_ids': batch_tokens.cpu().detach().numpy(),
                                'position_ids': position_ids.cpu().detach().numpy()}

                def have_attention_mask(sess):
                    return any(inp.name == 'attention_mask' for inp in all_inputs)

                if have_attention_mask(model.onnxrt):
                    tensor_dict['attention_mask'] = attention_mask.cpu().detach().numpy()
                torch.cuda.nvtx.range_pop() # "I/O Setup"
                output = model.onnxrt.run(['logits'], tensor_dict)[0]
                output = torch.Tensor(output).cuda()
                # output logits: [BS, SEQ, 50304]
            else:
                # nemo path
                torch.cuda.nvtx.range_pop() # "I/O Setup"
                output = inference_strategy.forward_step(batch, tensor_shape)
                output = output[0]['logits'].float()

            assert output is not None
            torch.cuda.nvtx.range_push("Output Sampling")
            output = output.float()
            logits = output[:, -1].view(batch_size, -1).contiguous()

            # make sure it will generate at least min_length
            min_length = extra.get('min_tokens_to_generate', 0)
            if min_length > 0:
                within_min_length = (context_length - context_lengths) < min_length
                logits[within_min_length, eod_id] = -float('Inf')

            # make sure it won't sample outside the vocab_size range
            logits[:, tokenizer.vocab_size :] = -float('Inf')

            # started indicates whether the current token step passes the context_length, so we make sure not to overwrite the context tokens
            started = context_lengths_cpu <= context_length
            if extra.get('greedy', False):
                prev = torch.argmax(logits, dim=-1).view(-1)
            else:
                logits = logits.float()
                logits /= temperature
                # handle repetition penality
                logits = repetition_penalty(logits, extra.get('repetition_penalty', 1.0), all_generated_indices)
                logits = top_k_logits(
                    logits, top_k=extra.get('top_k', 0), top_p=extra.get('top_p', 0.9), started=started
                )
                probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(probs, num_samples=1).view(-1)

            prev = prev.cpu()
            # Clamp the predicted out of vocabulary tokens
            prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
            # Replace sampled tokens w/ done token if EOD has already been sampled
            new_tokens = torch.where(is_done, eod_id, prev)
            # post process the inference tokens based on the strategy
            inference_strategy.post_process(tokens, new_tokens, context_length)

            # Insert either new predicted or next prompt token
            if extra.get("accuracy_mode", False):
                # We only update the last token for accuracy mode.
                at_prediction_index = (context_lengths + tokens_to_generate - 1 == context_length)
                tokens[:, context_length] = torch.where(at_prediction_index, new_tokens.cuda(), tokens[:, context_length])
            else:
                tokens[:, context_length] = torch.where(started.cuda(), new_tokens.cuda(), tokens[:, context_length])

            if not extra.get("benchmark_mode", False):
                if output_logits is None:
                    output = F.log_softmax(output[:, :context_length, :], 2)
                    indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                    output_logits = torch.gather(output, 2, indices).squeeze(2)
                    all_generated_indices = indices[:, :, 0]
                    if all_probs:
                        full_logits = output
                else:
                    output = F.log_softmax(output, 2)
                    indices = torch.unsqueeze(new_tokens.cuda(), 1).unsqueeze(2)
                    new_output_logits = torch.gather(output, 2, indices).squeeze(2)

                    # This copy can be optimized out by pre-allocating the memory.
                    output_logits = torch.cat([output_logits, new_output_logits], 1)
                    all_generated_indices = torch.cat([all_generated_indices, indices[:, :, 0]], 1)
                    if all_probs:
                        if extra.get("use_cache", False):
                            full_logits = torch.cat([full_logits, output], 1)
                        else:
                            full_logits = output

            done_token = (prev == eod_id)
            done_token = done_token.byte() & started.byte()

            just_finished = (done_token & ~is_done).bool()
            lengths[just_finished.view(-1)] = context_length
            is_done = is_done | done_token

            done = torch.all(is_done)
            torch.cuda.nvtx.range_pop() # "Output Sampling"

            context_length += 1
            counter += 1
            if done and not extra.get("benchmark_mode", False):
                break

        if all_probs:
            return tokens, context_length, lengths, output_logits, full_logits
        return tokens, context_length, lengths, output_logits, None

def initialize_ddp(model, cfg):
    # check whether the DDP is initialized
    if cfg.runtime == "nemo" and parallel_state.is_unitialized():
        def dummy():
            return
        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

        if model.cfg.get('transformer_engine', False):
            model.setup_transformer_engine_tp_groups()

def get_special_tokens(tokenizer):
    special_tokens = set()
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
        special_tokens.add(tokenizer.pad_token)
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
        special_tokens.add(tokenizer.eos_token)
    if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token is not None:
        special_tokens.add(tokenizer.bos_token)
    if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token is not None:
        special_tokens.add(tokenizer.cls_token)
    if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
        special_tokens.add(tokenizer.unk_token)
    if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token is not None:
        special_tokens.add(tokenizer.sep_token)
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
        special_tokens.add(tokenizer.mask_token)
    return special_tokens

def process_output(model, output, return_segments=False):
    torch.cuda.nvtx.range_push("Process Output")
    inference_strategy = GPTModelTextGenerationStrategy(model)
    tokenizer = model.tokenizer
    if output is not None:
        decode_tokens, output_logits, full_logits = output
        decode_tokens = decode_tokens.cpu().numpy().tolist()

        # convert ids to text by applying tokenizer
        resp_sentences = list(map(tokenizer.ids_to_text, decode_tokens))

        all_offsets = []
        resp_sentences_seg = []
        if return_segments:
            # segments sentences into words.
            for decode_token in decode_tokens:
                words = []
                for token in decode_token:
                    if not isinstance(token, Iterable):
                        token = [token]
                    word = tokenizer.ids_to_tokens(token)
                    if isinstance(word, Iterable):
                        word = word[0]
                    if hasattr(tokenizer.tokenizer, 'byte_decoder'):
                        word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                            'utf-8', errors='replace'
                        )
                    words.append(word)
                resp_sentences_seg.append(words)

            # offsets calculation
            special_tokens = get_special_tokens(tokenizer)
            for item in resp_sentences_seg:
                offsets = [0]
                for index, token in enumerate(item):
                    if index != len(item) - 1:
                        if token in special_tokens:
                            offsets.append(offsets[-1])
                        else:
                            offsets.append(len(token) + offsets[-1])
                all_offsets.append(offsets)

        output = {}
        output['sentences'] = resp_sentences
        output['tokens'] = resp_sentences_seg
        output['logprob'] = output_logits
        output['full_logprob'] = full_logits
        output['token_ids'] = decode_tokens
        output['offsets'] = all_offsets
        output = inference_strategy.post_generation_process(output)
    torch.cuda.nvtx.range_pop() # "Process Output"
    return output

def generate(model, inputs, cfg):
    torch.cuda.nvtx.range_push("Prepare Batch")
    initialize_ddp(model, cfg)

    tokens_to_generate = cfg.inference.tokens_to_generate
    min_tokens_to_generate = cfg.inference.min_tokens_to_generate
    add_BOS = cfg.inference.add_BOS
    all_probs = cfg.inference.all_probs
    temperature = cfg.inference.temperature
    is_benchmark_mode = True if cfg.mode == "benchmark" else False
    is_accuracy_mode = True if cfg.mode == "accuracy" else False

    inference_strategy = GPTModelTextGenerationStrategy(model)
    if isinstance(inputs, tuple):
        context_tokens_tensor, context_length_tensor = inputs
    else:
        context_tokens_tensor, context_length_tensor = inference_strategy.tokenize_batch(
            inputs, tokens_to_generate, add_BOS
        )

    context_length = context_length_tensor.min().item()

    batch_token_result = sample_sequence_batch(
        model,
        inference_strategy,
        context_tokens_tensor,
        context_length_tensor,
        tokens_to_generate,
        all_probs,
        temperature=temperature,
        extra={
            "top_p": cfg.inference.top_p,
            "top_k": cfg.inference.top_k,
            "greedy": cfg.inference.greedy,
            "repetition_penalty": cfg.inference.repetition_penalty,
            "min_tokens_to_generate": min_tokens_to_generate,
            "use_cache": cfg.use_cache,
            "benchmark_mode": is_benchmark_mode,
            "accuracy_mode": is_accuracy_mode,
            "use_fp8_storage": cfg.onnx_export_options.use_fp8_storage,
        },
    )

    tokens, context_length, _, output_logits, full_logits = batch_token_result

    output = None
    if tokens is not None:
        output = tokens[:, :context_length], output_logits, full_logits
    return output

def full_inference(model, inputs, cfg):
    output = generate(model, inputs, cfg)
    if output is not None:
        output = process_output(model, output, return_segments=(cfg.mode is not "benchmark"))
    return output
