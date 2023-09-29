# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import LlamaForCausalLM


__all__ = ["load_opt_model", "load_llama_model"]


def load_opt_model(model_path: str) -> torch.nn.Module:
    """
    Load a pretrained OPT model from the specified hugging face path

    :param model_path: hugging face path to model
    :return: loaded pretrained model
    """

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM

    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.seqlen = model.config.max_position_embeddings
    return model


def load_llama_model(model_path: str) -> torch.nn.Module:
    """
    Load a pretrained Llama model from the specified hugging face path

    :param model_path: hugging face path to model
    :return: loaded pretrained model
    """
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.eval()
    model.seqlen = model.config.max_position_embeddings
    return model
