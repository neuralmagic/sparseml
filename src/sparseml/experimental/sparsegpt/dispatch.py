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

SUPPORTED_MODELS = ["opt", "mpt", "llama"]


def load_model(args, model_key: str = None, *gargs, **kwargs):
    model_key = _get_model_key(args) if model_key is None else model_key
    if model_key == "opt":
        from sparseml.experimental.sparsegpt.opt import load_model as _load_model
    elif model_key == "mpt":
        from sparseml.experimental.sparsegpt.mpt import load_model as _load_model
    elif model_key == "llama":
        from sparseml.experimental.sparsegpt.llama2 import load_model as _load_model
    else:
        raise ValueError(f"Unrecognized model key. Supported: {SUPPORTED_MODELS}")
    return _load_model(args, *gargs, **kwargs)


def load_data(args, model_key: str = None, *gargs, **kwargs):
    model_key = _get_model_key(args) if model_key is None else model_key
    if model_key == "opt":
        from sparseml.experimental.sparsegpt.opt import load_data as _load_data
    elif model_key == "mpt":
        from sparseml.experimental.sparsegpt.mpt import load_data as _load_data
    elif model_key == "llama":
        from sparseml.experimental.sparsegpt.llama2 import load_data as _load_data
    else:
        raise ValueError(f"Unrecognized model key. Supported: {SUPPORTED_MODELS}")
    return _load_data(args, *gargs, **kwargs)


def evaluate_perplexity(
    args, model, dataloader, dev, model_key: str = None, *gargs, **kwargs
):
    model_key = _get_model_key(args) if model_key is None else model_key
    if model_key == "opt":
        from sparseml.experimental.sparsegpt.opt import ppl_eval as _ppl_eval
    elif model_key == "llama":
        from sparseml.experimental.sparsegpt.llama2 import ppl_eval as _ppl_eval
    else:
        raise ValueError(f"Unrecognized model key. Supported: {SUPPORTED_MODELS}")
    return _ppl_eval(args, model, dataloader, dev, *gargs, **kwargs)


def prepare_sparsegpt(model, dataloader, args, model_key: str = None, **kwargs):
    model_key = _get_model_key(args) if model_key is None else model_key
    if model_key == "opt":
        from sparseml.experimental.sparsegpt.opt import (
            prepare_sparsegpt as _prepare_sparsegpt,
        )
    elif model_key == "mpt":
        from sparseml.experimental.sparsegpt.mpt import (
            prepare_sparsegpt as _prepare_sparsegpt,
        )
    elif model_key == "llama":
        from sparseml.experimental.sparsegpt.llama2 import (
            prepare_sparsegpt as _prepare_sparsegpt,
        )
    else:
        raise ValueError(f"Unrecognized model key. Supported: {SUPPORTED_MODELS}")
    return _prepare_sparsegpt(model, dataloader, args, **kwargs)


def _get_model_key(args):
    key = None
    for k in SUPPORTED_MODELS:
        if args.model.lower().find(k) >= 0:
            key = k
            break
    if key is None:
        raise ValueError(
            f"Model {args.model} is not supported. Supported models: {SUPPORTED_MODELS}"
        )
    return key
