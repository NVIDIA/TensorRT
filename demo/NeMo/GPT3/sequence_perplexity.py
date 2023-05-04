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

import math
import numpy as np
import torch

__all__ = ['SequencePerplexity']

class SequencePerplexity():
    def __init__(self, topN):
        super().__init__()
        self.ppls = []
        self.sequence_ppls = []
        self.topN_equals = [0] * len(topN)
        self.topN = topN

    def update(self, ds_input, response, tokenizer):
        for batch, tokens in enumerate(response['token_ids']):
            inp_len = ds_input.lens[batch]
            if inp_len == 0:
                continue

            conti_len = ds_input.conti_len[batch]

            response_token_ids = tokens[:inp_len]
            assert response_token_ids == ds_input.inp_enc[batch][:-1], f"Mismatch in input tokens."
            full_log_probs = response['full_logprob'][batch][:inp_len]

            # calculate ppl with whole sequence.
            label = torch.tensor([ds_input.inp_enc[batch][1:]]).cuda()
            log_probs = full_log_probs.unsqueeze(0).permute((0, 2, 1))
            ppl = torch.nn.CrossEntropyLoss()(log_probs, label)
            self.sequence_ppls.append(ppl.cpu())

            # calculate topN.
            log_probs = full_log_probs[-conti_len:]
            conti_token_ids = ds_input.inp_enc[batch][-conti_len:]
            conti_tokens = tokenizer.ids_to_tokens(conti_token_ids)

            for index, topN in enumerate(self.topN):
                if conti_token_ids[0] in log_probs.topk(topN, dim=-1).indices:
                    self.topN_equals[index] += 1 

            # calculate ppl with last token.
            log_probs = log_probs.cpu().to(torch.float32)
            conti_enc = torch.tensor(tokenizer.tokens_to_ids(conti_tokens))
            conti_probs = torch.gather(log_probs, 1, conti_enc.unsqueeze(-1)).squeeze(-1)

            ppl = float(conti_probs.sum())
            self.ppls.append(ppl)

    def compute(self):
        ppls = math.exp(-np.mean(np.array(self.ppls)))
        sequence_ppls = math.exp(np.mean(np.array(self.sequence_ppls)))
        acc = [equals / len(self.ppls) for equals in self.topN_equals]
        txt = []
        for i, j in zip(self.topN, acc):
            txt.append("acc(top{}): {:.4f}".format(i, j))
        acc_text = ", ".join(txt)
        return ppls, sequence_ppls, acc, acc_text

