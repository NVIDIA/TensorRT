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

import gc
import os
import sys

# Only print out error messages from NeMo
from nemo.utils.nemo_logging import Logger as NG_LOGGER
nemo_logger = NG_LOGGER(False)
nemo_logger.setLevel(nemo_logger.ERROR)

from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
import torch

sys.path.append('../../HuggingFace') # Include HuggingFace directory.
from NNDF.logger import G_LOGGER


def get_computeprob_response(tokenizer, response, inputs):
    """
        This function is a modified version from:
        https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/modules/common/text_generation_utils.py#L139

        So parallel state does not need to be initialized before calling this function.
    """
    compute_prob_response = {}
    new_token_ids = []
    new_tokens = []
    new_texts = []
    log_probs = []
    full_logprobs = []
    offsets = []
    for batch_id in range(len(response['tokens'])):
        if isinstance(inputs, (list, tuple)):
            if isinstance(inputs[0], str):
                new_token_id = tokenizer.text_to_ids(inputs[batch_id])
                new_text = inputs[batch_id]
                token_len = len(new_token_id)
            elif isinstance(inputs[0], torch.Tensor):
                token_len = int(inputs[1][batch_id].item())
                new_token_id = inputs[0][batch_id][:token_len].tolist()
                new_text = tokenizer.ids_to_text(new_token_id)
        new_token_ids.append(new_token_id)
        new_tokens.append(response['tokens'][batch_id][:token_len])
        new_texts.append(new_text)
        log_probs.append(response['logprob'][batch_id][:token_len])
        full_logprobs.append(response['full_logprob'][batch_id][:token_len])
        offsets.append(response['offsets'][batch_id][:-1])
    compute_prob_response['sentences'] = new_texts
    compute_prob_response['tokens'] = new_tokens
    compute_prob_response['token_ids'] = new_token_ids
    compute_prob_response['logprob'] = log_probs
    compute_prob_response['full_logprob'] = full_logprobs
    compute_prob_response['offsets'] = offsets
    return compute_prob_response


def load_nemo_model(cfg, model_class=MegatronGPTModel):
    # Trainer is required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)

    if cfg.gpt_model_file and cfg.checkpoint_dir:
        raise ValueError(f"NeMo model and checkpoint cannot be both set.")

    if cfg.gpt_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.gpt_model_file

        pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.sequence_parallel = False
            pretrained_cfg.activations_checkpoint_granularity = None
            pretrained_cfg.activations_checkpoint_method = None
            pretrained_cfg.precision = trainer.precision
            if trainer.precision == "16":
                pretrained_cfg.megatron_amp_O2 = False
        model = model_class.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
        )
        G_LOGGER.info(f"{type(model)} has been successfully restored from {cfg.gpt_model_file}")
    elif cfg.checkpoint_dir:
        checkpoint_file= os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        if not os.path.exists(checkpoint_file):
            raise ValueError(f"File {checkpoint_file} does not exist.")

        app_state = AppState()
        if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
            app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            (
                app_state.tensor_model_parallel_rank,
                app_state.pipeline_model_parallel_rank,
                app_state.model_parallel_size,
                app_state.data_parallel_size,
                app_state.pipeline_model_parallel_split_rank,
                app_state.virtual_pipeline_model_parallel_rank,
            ) = fake_initialize_model_parallel(
                world_size=app_state.model_parallel_size,
                rank=trainer.global_rank,
                tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
                pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
            )
        checkpoint_path = inject_model_parallel_rank(checkpoint_file)
        model = model_class.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
        G_LOGGER.info(f"{type(model)} has been successfully restored from checkpoint {checkpoint_path}")
    else:
        raise ValueError("Need to provide a nemo gpt model through config file.")

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    model.eval()
    G_LOGGER.debug(f"Model configuration: {model.cfg}")
    G_LOGGER.debug(f"Vocabulary size: {model.tokenizer.vocab_size}")
    return model.cuda()

def release_nemo_model(model):
    print(f"Releaseing nemo model.")
    model.model.cpu()
    del model.model
    gc.collect()
    torch.cuda.empty_cache()
    model.model = None
