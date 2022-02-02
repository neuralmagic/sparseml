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

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.nn import Module
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.file_utils import WEIGHTS_NAME

from sparseml.pytorch.utils import get_prunable_layers, tensor_sparsity


__all__ = ["SparseAutoModel"]


_LOGGER = logging.getLogger(__name__)


class SparseAutoModel:
    """
    Factory class for creating sparse models using transformers AutoModel classes
    """

    @staticmethod
    def masked_language_modeling_from_pretrained(
        model_name_or_path: str,
        model_type: str,
        config: Any,
        **kwargs,
    ) -> Module:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param model_type: specify the type of model loaded for logging;
            ex one of [model, student, teacher]
        :param config: the config for the model describing pipeline
        :param kwargs: keyword arguments to pass through to the AutoModel call
        :return: the created model for masked language modeling
        """
        delayed = False
        if not model_name_or_path:
            _LOGGER.info("Training new model from scratch")
            model = AutoModelForMaskedLM.from_config(config)
        else:
            SparseAutoModel._check_tf(model_name_or_path)
            if not kwargs:
                kwargs = {}
            kwargs["from_tf"] = False
            if "state_dict" not in kwargs:
                kwargs["state_dict"], delayed = SparseAutoModel._loadable_state_dict(
                    model_name_or_path
                )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                **kwargs,
            )

        SparseAutoModel.log_model_load(model, model_name_or_path, model_type, delayed)

        return model

    @staticmethod
    def masked_language_modeling_from_pretrained_distil(
        model_name_or_path: str,
        teacher_name_or_path: Optional[str],
        model_kwargs: Dict[str, Any],
        teacher_kwargs: Dict[str, Any],
    ) -> Tuple[Module, Optional[Union[Module, str]]]:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param teacher_name_or_path: the name of or path to the teacher to load,
            None or one of ['self', 'disable'] will not create a teacher and
            instead return the value passed in
        :param model_kwargs: the keyword args to pass into the AutoModel for model
        :param teacher_kwargs: the keyword args to pass into the AutoModel for teacher
        :return: a tuple containing the model and distillation teacher (optional)
            for masked language modeling
        """
        model = SparseAutoModel.masked_language_modeling_from_pretrained(
            model_name_or_path,
            model_type="student" if teacher_name_or_path else "model",
            config=model_kwargs["config"],
            **model_kwargs,
        )
        teacher = (
            SparseAutoModel.masked_language_modeling_from_pretrained(
                teacher_name_or_path,
                model_type="teacher",
                config=None,
                **teacher_kwargs,
            )
            if teacher_name_or_path and teacher_name_or_path not in ["self", "disable"]
            else teacher_name_or_path
        )

        return model, teacher

    @staticmethod
    def question_answering_from_pretrained(
        model_name_or_path: str,
        model_type: str,
        **kwargs,
    ) -> Module:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param model_type: specify the type of model loaded for logging;
            ex one of [model, student, teacher]
        :param kwargs: keyword arguments to pass through to the AutoModel call
        :return: the created model for question answering
        """
        SparseAutoModel._check_tf(model_name_or_path)
        if not kwargs:
            kwargs = {}
        kwargs["from_tf"] = False
        delayed = False
        if "state_dict" not in kwargs:
            kwargs["state_dict"], delayed = SparseAutoModel._loadable_state_dict(
                model_name_or_path
            )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
        SparseAutoModel.log_model_load(model, model_name_or_path, model_type, delayed)

        return model

    @staticmethod
    def question_answering_from_pretrained_distil(
        model_name_or_path: str,
        teacher_name_or_path: Optional[str],
        model_kwargs: Dict[str, Any],
        teacher_kwargs: Dict[str, Any],
    ) -> Tuple[Module, Optional[Union[Module, str]]]:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param teacher_name_or_path: the name of or path to the teacher to load,
            None or one of ['self', 'disable'] will not create a teacher and
            instead return the value passed in
        :param model_kwargs: the keyword args to pass into the AutoModel for model
        :param teacher_kwargs: the keyword args to pass into the AutoModel for teacher
        :return: a tuple containing the model and distillation teacher (optional)
            for question answering
        """
        model = SparseAutoModel.question_answering_from_pretrained(
            model_name_or_path,
            model_type="student" if teacher_name_or_path else "model",
            **model_kwargs,
        )
        teacher = (
            SparseAutoModel.question_answering_from_pretrained(
                teacher_name_or_path, model_type="teacher", **teacher_kwargs
            )
            if teacher_name_or_path and teacher_name_or_path not in ["self", "disable"]
            else teacher_name_or_path
        )

        return model, teacher

    @staticmethod
    def text_classification_from_pretrained(
        model_name_or_path: str,
        model_type: str,
        **kwargs,
    ) -> Module:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param model_type: specify the type of model loaded for logging;
            ex one of [model, student, teacher]
        :param kwargs: keyword arguments to pass through to the AutoModel call
        :return: the created model for text classification
        """
        SparseAutoModel._check_tf(model_name_or_path)
        if not kwargs:
            kwargs = {}
        kwargs["from_tf"] = False
        delayed = False
        if "state_dict" not in kwargs:
            kwargs["state_dict"], delayed = SparseAutoModel._loadable_state_dict(
                model_name_or_path
            )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
        SparseAutoModel.log_model_load(model, model_name_or_path, model_type, delayed)

        return model

    @staticmethod
    def text_classification_from_pretrained_distil(
        model_name_or_path: str,
        teacher_name_or_path: Optional[str],
        model_kwargs: Dict[str, Any],
        teacher_kwargs: Dict[str, Any],
    ) -> Tuple[Module, Optional[Module]]:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param teacher_name_or_path: the name of or path to the teacher to load,
            None or one of ['self', 'disable'] will not create a teacher and
            instead return the value passed in
        :param model_kwargs: the keyword args to pass into the AutoModel for model
        :param teacher_kwargs: the keyword args to pass into the AutoModel for teacher
        :return: a tuple containing the model and distillation teacher (optional)
            for sequence/text classification
        """
        model = SparseAutoModel.text_classification_from_pretrained(
            model_name_or_path,
            model_type="student" if teacher_name_or_path else "model",
            **model_kwargs,
        )
        teacher = (
            SparseAutoModel.text_classification_from_pretrained(
                teacher_name_or_path, model_type="teacher", **teacher_kwargs
            )
            if teacher_name_or_path and teacher_name_or_path not in ["self", "disable"]
            else teacher_name_or_path
        )

        return model, teacher

    @staticmethod
    def token_classification_from_pretrained(
        model_name_or_path: str,
        model_type: str,
        **kwargs,
    ) -> Module:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param model_type: specify the type of model loaded for logging;
            ex one of [model, student, teacher]
        :param kwargs: keyword arguments to pass through to the AutoModel call
        :return: the created model for token classification
        """
        SparseAutoModel._check_tf(model_name_or_path)
        if not kwargs:
            kwargs = {}
        kwargs["from_tf"] = False
        delayed = False
        if "state_dict" not in kwargs:
            kwargs["state_dict"] = SparseAutoModel._loadable_state_dict(
                model_name_or_path
            )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
        SparseAutoModel.log_model_load(model, model_name_or_path, model_type, delayed)

        return model

    @staticmethod
    def token_classification_from_pretrained_distil(
        model_name_or_path: str,
        teacher_name_or_path: Optional[str],
        model_kwargs: Dict[str, Any],
        teacher_kwargs: Dict[str, Any],
    ) -> Tuple[Module, Optional[Module]]:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param teacher_name_or_path: the name of or path to the teacher to load,
            None or one of ['self', 'disable'] will not create a teacher and
            instead return the value passed in
        :param model_kwargs: the keyword args to pass into the AutoModel for model
        :param teacher_kwargs: the keyword args to pass into the AutoModel for teacher
        :return: a tuple containing the model and distillation teacher (optional)
            for token classification
        """
        model = SparseAutoModel.token_classification_from_pretrained(
            model_name_or_path,
            model_type="student" if teacher_name_or_path else "model",
            **model_kwargs,
        )
        teacher = (
            SparseAutoModel.token_classification_from_pretrained(
                teacher_name_or_path, model_type="teacher", **teacher_kwargs
            )
            if teacher_name_or_path and teacher_name_or_path not in ["self", "disable"]
            else teacher_name_or_path
        )

        return model, teacher

    @staticmethod
    def log_model_load(
        model: Module, model_name_or_path: str, model_type: str, delayed_load: bool
    ):
        """
        Log the state of a loaded model including sparsity and
        prunable params information.

        :param model: the loaded model
        :param model_name_or_path: the original name of or path to the model that loaded
        :param model_type: specify the type of model loaded for logging;
            ex one of [model, student, teacher]
        :param delayed_load: True if this model load was delayed until after
            recipe instantiation due to QAT or other architectural state changes
        """
        if delayed_load:
            _LOGGER.info(
                f"Delayed load of model {model_name_or_path} detected. "
                f"Will print out model information once SparseML recipes have loaded"
            )
            return

        model_params = list(
            filter(lambda param: param.requires_grad, model.parameters())
        )
        total_params = sum(torch.numel(param) for param in model_params)
        params_info = {
            f"{name}.weight": {
                "sparsity": tensor_sparsity(layer.weight).item(),
                "numel": torch.numel(layer.weight),
            }
            for (name, layer) in get_prunable_layers(model)
        }
        prunable_sparse_params = sum(
            round(param["numel"] * param["sparsity"]) for param in params_info.values()
        )
        prunable_total_params = sum(
            round(param["numel"]) for param in params_info.values()
        )
        avg_prunable_sparsity = float(prunable_sparse_params) / prunable_total_params

        _LOGGER.info(
            f"Loaded {model_type} from {model_name_or_path} "
            f"with {total_params} total params. "
            f"Of those there are {prunable_total_params} prunable params "
            f"which have {avg_prunable_sparsity} avg sparsity."
        )
        _LOGGER.info(
            f"{'sparse' if avg_prunable_sparsity > 0.05 else 'dense'} model detected, "
            f"prunable params info: {json.dumps(params_info)}"
        )

    @staticmethod
    def _loadable_state_dict(
        model_name_or_path: str,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        :param model_name_or_path: name of or path to model
        :return: (loaded state dict, True if overriding state dict for delayed load)
            delayed load happens when a QAT graph is detected since a recipe
            must be applied first
        """
        if not model_name_or_path or not os.path.isfile(
            os.path.join(model_name_or_path, WEIGHTS_NAME)
        ):
            return None, False

        state_dict = torch.load(
            os.path.join(model_name_or_path, WEIGHTS_NAME), map_location="cpu"
        )
        is_qat_state = any(
            [
                key.endswith(".zero_point") or key.endswith(".observer_enabled")
                for key in state_dict.keys()
            ]
        )

        if not is_qat_state:
            return None, False

        _LOGGER.warning(
            "QAT state detected, ignore any loading errors, weights will reload "
            f"after SparseML recipes have been applied {model_name_or_path}"
        )

        return {}, True

    @staticmethod
    def _check_tf(model_name_or_path: str):
        if ".ckpt" in model_name_or_path:
            raise ValueError(
                "PyTorch is the only supported model type currently for SparseML "
                "HuggingFace Transformers integration. "
                "Detected a TensorFlow model from model_name_or_path: "
                f"{model_name_or_path}"
            )
