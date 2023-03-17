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

import numpy as np
import nltk
from nltk import word_tokenize
import json
import tensorrt as trt


def preprocess(text):
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    tokens = word_tokenize(text)
    # split into lower-case word tokens, in numpy array with shape of (seq, 1)
    words = np.asarray([w.lower() for w in tokens]).reshape(-1, 1)
    # split words into chars, in numpy array with shape of (seq, 1, 1, 16)
    chars = [[c for c in t][:16] for t in tokens]
    chars = [cs + [""] * (16 - len(cs)) for cs in chars]
    chars = np.asarray(chars).reshape(-1, 1, 1, 16)
    return words, chars


def get_map_func(filepath):
    file = open(filepath)
    category_map = json.load(file)
    category_mapper = dict(zip(category_map["cats_strings"], category_map["cats_int64s"]))
    default_int64 = category_map["default_int64"]
    func = lambda s: category_mapper.get(s, default_int64)
    return np.vectorize(func)


def get_inputs(context, query):
    cw, cc = preprocess(context)
    qw, qc = preprocess(query)

    context_word_func = get_map_func("CategoryMapper_4.json")
    context_char_func = get_map_func("CategoryMapper_5.json")
    query_word_func = get_map_func("CategoryMapper_6.json")
    query_char_func = get_map_func("CategoryMapper_7.json")

    cw_input = context_word_func(cw).astype(trt.nptype(trt.int32)).ravel()
    cc_input = context_char_func(cc).astype(trt.nptype(trt.int32)).ravel()
    qw_input = query_word_func(qw).astype(trt.nptype(trt.int32)).ravel()
    qc_input = query_char_func(qc).astype(trt.nptype(trt.int32)).ravel()
    return cw_input, cc_input, qw_input, qc_input
