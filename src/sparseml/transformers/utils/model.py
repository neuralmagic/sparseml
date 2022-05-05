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

import inspect
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

from sparseml.pytorch.utils import ModuleSparsificationInfo


__all__ = ["SparseAutoModel", "get_shared_tokenizer_src"]


_LOGGER = logging.getLogger(__name__)


class SparseAutoModel:
    """
    Factory class for creating sparse models using transformers AutoModel classes
    """

    @staticmethod
    def masked_language_modeling_from_pretrained(
        model_name_or_path: str,
        model_type: str,
        **kwargs,
    ) -> Module:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param model_type: specify the type of model loaded for logging;
            ex one of [model, student, teacher]
        :param kwargs: keyword arguments to pass through to the AutoModel call
        :return: the created model for masked language modeling
        """
        delayed = False
        if not model_name_or_path:
            _LOGGER.info("Training new model from scratch")
            config = kwargs["config"]
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
            model = AutoModelForMaskedLM.from_pretrained(
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
            **model_kwargs,
        )
        teacher = (
            SparseAutoModel.masked_language_modeling_from_pretrained(
                teacher_name_or_path,
                model_type="teacher",
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
            kwargs["state_dict"], delayed = SparseAutoModel._loadable_state_dict(
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

        sparsification_info = ModuleSparsificationInfo(model)

        _LOGGER.info(
            f"Loaded {model_type} from {model_name_or_path} "
            f"with {sparsification_info.params_total} total params. "
            f"Of those there are {sparsification_info.params_prunable_total} prunable "
            f"params which have {sparsification_info.params_prunable_sparse_percent} "
            "avg sparsity."
        )
        model_type = (
            "sparse"
            if sparsification_info.params_prunable_sparse_percent > 5
            else "dense"
        )
        _LOGGER.info(
            f"{model_type} model detected, "
            f"all sparsification info: {sparsification_info}"
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


def get_shared_tokenizer_src(student: Module, teacher: Optional[Module]) -> str:
    """
    Get a tokenizer source used for both student and teacher, assuming
    that they could be shared

    :param student: the student model
    :param teacher: the teacher model
    :return: the source for the tokenizer shared between teacher and model
    """

    if teacher is not None and teacher not in ("disable", "self"):
        student_forward_params = list(
            inspect.signature(student.forward).parameters.keys()
        )
        teacher_forward_params = list(
            inspect.signature(teacher.forward).parameters.keys()
        )
        diff = [p for p in student_forward_params if p not in teacher_forward_params]
        if diff:
            raise RuntimeError(
                "Teacher tokenizer cannot be used for student "
                f"due to missing args: {diff}"
            )
        src_model = teacher
    else:
        src_model = student
    return src_model.config._name_or_path
