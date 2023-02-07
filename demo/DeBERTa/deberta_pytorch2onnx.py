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

'''
Generate HuggingFace DeBERTa (V2) model with different configurations (e.g., sequence length, hidden size, No. of layers, No. of heads, etc.) and export in ONNX format

Usage:
    python deberta_pytorch2onnx.py [--filename xx.onnx] [--variant microsoft/deberta-xx] [--seq-len xx]
'''

import os, time, argparse 
from transformers import DebertaV2Tokenizer, DebertaV2Config, DebertaV2ForSequenceClassification 
# DEBERTA V2 implementation, https://github.com/huggingface/transformers/blob/master/src/transformers/models/deberta_v2/modeling_deberta_v2.py
import torch, onnxruntime as ort, numpy as np

parser = argparse.ArgumentParser(description="Generate HuggingFace DeBERTa (V2) model with different configurations and export in ONNX format. This will save the model under the same directory as 'deberta_seqxxx_hf.onnx'.")
parser.add_argument('--filename', type=str, help='Path to the save the ONNX model')
parser.add_argument('--variant', type=str, default=None, help='DeBERTa variant name. Such as microsoft/deberta-v3-xsmall')
parser.add_argument('--seq-len', type=int, default=None, help='Specify maximum sequence length. Note: --variant and --seq-len cannot be used together. Pre-trained models have pre-defined sequence length')

args = parser.parse_args()
onnx_filename = args.filename
model_variant = args.variant 
sequence_length = args.seq_len

assert not args.variant or (args.variant and not args.seq_len), "--variant and --seq-len cannot be used together!"
assert torch.cuda.is_available(), "CUDA not available!"

def randomize_model(model):
    for module_ in model.named_modules(): 
        if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
            module_[1].weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        elif isinstance(module_[1], torch.nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)
        if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()
    return model

def export():
    parent_dir = os.path.dirname(onnx_filename)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        
    if model_variant is None:
        # default model hyper-params
        batch_size = 1
        seq_len = 2048 if sequence_length is None else sequence_length
        max_position_embeddings = 512 if seq_len <= 512 else seq_len # maximum sequence length that this model might ever be used with. By default 512. otherwise error https://github.com/huggingface/transformers/issues/4542
        vocab_size = 128203
        hidden_size = 384
        layers = 12
        heads = 6
        intermediate_size = hidden_size*4 # feed forward layer dimension
        type_vocab_size = 0
        # relative attention
        relative_attention=True
        max_relative_positions = 256 # k
        pos_att_type = ["p2c", "c2p"]
    
        deberta_config = DebertaV2Config(vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=layers, num_attention_heads=heads, intermediate_size=intermediate_size, type_vocab_size=type_vocab_size, max_position_embeddings=max_position_embeddings, relative_attention=relative_attention, max_relative_positions=max_relative_positions, pos_att_type=pos_att_type)
        deberta_model = DebertaV2ForSequenceClassification(deberta_config)
        deberta_model = randomize_model(deberta_model)
    else:
        deberta_model = DebertaV2ForSequenceClassification.from_pretrained(model_variant)
        deberta_config = DebertaV2Config.from_pretrained(model_variant)

        batch_size = 1
        seq_len = deberta_config.max_position_embeddings
        vocab_size = deberta_config.vocab_size
    
    deberta_model.cuda().eval()

    # input/output
    gpu = torch.device('cuda')
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=gpu)
    attention_mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long, device=gpu)
    input_names = ['input_ids', 'attention_mask']   
    output_names = ['output']
    dynamic_axes={'input_ids'   : {0 : 'batch_size'}, 
                  'attention_mask'   : {0 : 'batch_size'},   
                  'output' : {0 : 'batch_size'}}
    
    # ONNX export
    torch.onnx.export(deberta_model, # model 
                     (input_ids, attention_mask), # model inputs
                     onnx_filename,
                     export_params=True,
                     opset_version=13,
                     do_constant_folding=True,
                     input_names = input_names,
                     output_names = output_names,
                     dynamic_axes = dynamic_axes)
    
    # full precision inference
    num_trials = 10

    start = time.time()
    for i in range(num_trials):
        results = deberta_model(input_ids, attention_mask)
    end = time.time()

    print("Average PyTorch FP32(TF32) time: {:.2f} ms".format((end - start)/num_trials*1000))
    
    # half precision inference (do this after onnx export, otherwise the export ONNX model is with FP16 weights...)
    deberta_model_fp16 = deberta_model.half()
    start = time.time()
    for i in range(num_trials):
        results = deberta_model_fp16(input_ids, attention_mask)
    end = time.time()

    print("Average PyTorch FP16 time: {:.2f} ms".format((end - start)/num_trials*1000))

    # model size
    total_params = sum(param.numel() for param in deberta_model.parameters())
    print("Total # of params: ", total_params)
    print("Maximum sequence length: ", seq_len)

if __name__ == "__main__":
    export()
