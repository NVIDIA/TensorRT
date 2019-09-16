#!/usr/bin/env bash

# Copyright 2017 Google Inc.
# Modifications copyright (C) 2019 Nvidia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


ROOT_DIR="${1:-${PWD}}"
OUTPUT_DIR="${ROOT_DIR}/intermediate_data"
DATA_DIR="${ROOT_DIR}/data/nmt/deen"
SCRIPTS_DIR="${ROOT_DIR}/scripts"
BPE_CODES="${DATA_DIR}/bpe.32000"


sgm_to_txt(){
    ${SCRIPTS_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl < $1 > $2
}

tokenize(){
    lan=$1
    ${SCRIPTS_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l ${lan} -threads 8 < $2 > $3
}

split_subwords(){
    if [ ! -f "${BPE_CODES}" ]; then
        echo "ERROR: ${BPE_CODES} not found. A codes file is required to split newstest into subwords."
        exit 0
    fi
    ${SCRIPTS_DIR}/subword-nmt/subword_nmt/apply_bpe.py -c ${BPE_CODES} < $1 > $2
}

mkdir -p ${OUTPUT_DIR}
mkdir -p ${SCRIPTS_DIR}
mkdir -p ${DATA_DIR}

# TBD: download bpe and vocab.
if [ ! -f ${OUTPUT_DIR}/sampleNMT_data.tar.bz2 ]; then
    echo "Downloading sample_nmt support data..."
    curl -o ${OUTPUT_DIR}/sampleNMT_data.tar.bz2 \
        https://developer.download.nvidia.com/compute/machine-learning/tensorrt/models/sampleNMT_data.tar.bz2 
fi

echo "Extracting sample_nmt support data..."
tar -C ${DATA_DIR} -xf ${OUTPUT_DIR}/sampleNMT_data.tar.bz2


# Clone required scripts
# moses
if [ ! -d "${SCRIPTS_DIR}/mosesdecoder" ]; then
  echo "Cloning moses..."
  git clone https://github.com/moses-smt/mosesdecoder.git "${SCRIPTS_DIR}/mosesdecoder"
fi

# subword-nmt
if [ ! -d "${SCRIPTS_DIR}/subword-nmt" ]; then
  echo "Cloning subword-nmt..."
  git clone https://github.com/rsennrich/subword-nmt.git "${SCRIPTS_DIR}/subword-nmt"
fi


echo "Downloading newstests..."
curl -o ${OUTPUT_DIR}/dev.tgz \
  http://data.statmt.org/wmt16/translation-task/dev.tgz

# Extract everything
echo "Extracting newstest data..."
mkdir -p "${OUTPUT_DIR}/SGM"
tar -xzf "${OUTPUT_DIR}/dev.tgz" -C "${OUTPUT_DIR}/SGM"

# Convert newstest2015 SGM file into raw text format
echo "Converting newstest2015 SGM file into raw text format..."
raw_de=${OUTPUT_DIR}/newstest2015.de
raw_en=${OUTPUT_DIR}/newstest2015.en
sgm_to_txt ${OUTPUT_DIR}/SGM/dev/newstest2015-deen-src.de.sgm ${raw_de}
sgm_to_txt ${OUTPUT_DIR}/SGM/dev/newstest2015-deen-ref.en.sgm ${raw_en}

# Tokenize newstest files
echo "Tokenizing..."
tok_de=${OUTPUT_DIR}/newstest2015.tok.de
tok_en=${OUTPUT_DIR}/newstest2015.tok.en
tokenize de ${raw_de} ${tok_de}
tokenize en ${raw_en} ${tok_en}


# Split into subwords
echo "Splitting into subwords..."
bpe_de=${DATA_DIR}/newstest2015.tok.bpe.32000.de
bpe_en=${DATA_DIR}/newstest2015.tok.bpe.32000.en
split_subwords ${tok_de} ${bpe_de}
split_subwords ${tok_en} ${bpe_en}

