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
import os
import collections
import json
import requests
import sys
import torch
from torch.nn.utils.rnn import pad_sequence

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

from nemo_export import create_dir_if_not_exist

__all__ = ['Lambada']


class Lambada():

    def __init__(self, base_dir, tokens_to_generate, padding = -1, max_length = 2048):
        assert tokens_to_generate >= 1
        assert padding == -1 or tokens_to_generate == 1
        self.base_dir = base_dir
        self.tokens_to_generate = tokens_to_generate
        self.padding = padding
        self.max_length = max_length 
        self.download()

    def get_data_file_path(self):
        path = os.path.join(self.base_dir, "lambada")
        path = os.path.join(path, "lambada_test.jsonl")
        create_dir_if_not_exist(path)
        return path

    def download(self):
        path = self.get_data_file_path()
        if not os.path.exists(path):
            url = "https://github.com/cybertronai/bflm/raw/master/lambada_test.jsonl"
            with requests.get(url) as r, open(path, 'wb') as fh:
                fh.write(r.content)

    def load(self):
        path = self.get_data_file_path()
        with open(path) as fh:
            for line in fh:
                yield json.loads(line)

    def _preprocess(self, text):
        text = text.replace("“", '"')
        text = text.replace("”", '"')
        text = text.replace("’", "'")
        text = text.replace("‘", "'")
        return text

    def doc_to_text(self, doc):
        return "\n" + self._preprocess(doc["text"].rsplit(" ", 1)[0]).strip()

    def doc_to_target(self, doc):
        split_text = doc["text"].rsplit(" ", 1)
        if len(split_text) <= 1:
            raise ValueError(f"Input doc '{doc}' does not have target.")
        return " " + self._preprocess(split_text[1])

    def preprocess_input(self, tokenizer, docs):
        _Input = collections.namedtuple("_DS_Input", ["inputs", "inp_enc", "lens", "lens_pad", "conti_len"])
        batch_size = len(docs)
        tokens = []
        conti_lens = []
        lens = []
        inp_encs = []
        for doc in docs:
            # Handle padded text
            if not doc["text"]:
                inp_enc = [0]
                conti_len = 0
            else:
                text = self.doc_to_text(doc)
                target = self.doc_to_target(doc)

                context_enc = tokenizer.text_to_ids(text)
                continuation_enc = tokenizer.text_to_ids(target)

                inp_enc = (context_enc + continuation_enc)[-(self.max_length + 1) :]
                conti_len = len(continuation_enc)

            inp_encs.append(inp_enc)
            conti_lens.append(conti_len)
            tokens.append(torch.tensor(inp_enc))
            lens.append(len(inp_enc) - 1)
        max_lens = max(lens)

        tokens_pad = pad_sequence(tokens, batch_first=False, padding_value=tokenizer.eos_id)
        if self.padding != -1 and max_lens % self.padding != 0:
            # We need align the context length to multiple of 8 for FP8 run using NeMo framework.
            extra_pad_len = self.padding - (max_lens % self.padding)

            extra_pad = torch.ones(extra_pad_len, batch_size) * tokenizer.eos_id
            extra_pad = extra_pad.type_as(tokens_pad)
            inp_enc_pad = torch.vstack((tokens_pad, extra_pad)).T

            lens_pad = max_lens + extra_pad_len
        else:
            inp_enc_pad = tokens_pad.T
            lens_pad = max_lens + 1 - self.tokens_to_generate

        inputs = (torch.tensor(inp_enc_pad).cuda(), (torch.ones(batch_size, dtype=torch.int32) * lens_pad).cuda())
        return _Input(inputs=inputs, inp_enc=inp_encs, lens=lens, lens_pad=lens_pad, conti_len=conti_lens)

