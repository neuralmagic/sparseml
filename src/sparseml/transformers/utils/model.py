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

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy
import torch
from torch.nn import Module
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.file_utils import WEIGHTS_NAME


__all__ = ["SparseAutoModel"]


class SparseAutoModel:
    @staticmethod
    def masked_language_modeling_from_pretrained(
        model_name_or_path: str,
        config: Any,
        logger: Optional[logging.Logger],
        **kwargs,
    ) -> Module:
        if not model_name_or_path:
            logger.info("Training new model from scratch")
            return AutoModelForMaskedLM.from_config(config)

        SparseAutoModel._check_tf(model_name_or_path)
        if not kwargs:
            kwargs = {}
        kwargs["from_tf"] = False
        if "state_dict" not in kwargs:
            kwargs["state_dict"] = SparseAutoModel.loadable_state_dict(
                model_name_or_path, logger
            )

        return AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

    @staticmethod
    def masked_language_modeling_from_pretrained_distil(
        model_name_or_path: str,
        teacher_name_or_path: Optional[str],
        model_kwargs: Dict[str, Any],
        teacher_kwars: Dict[str, Any],
        logger: Optional[logging.Logger],
    ) -> Tuple[Module, Optional[Union[Module, str]]]:
        model = SparseAutoModel.masked_language_modeling_from_pretrained(
            model_name_or_path, model_kwargs["config"], logger, **model_kwargs
        )
        teacher = (
            SparseAutoModel.masked_language_modeling_from_pretrained(
                teacher_name_or_path, None, logger, **teacher_kwars
            )
            if teacher_name_or_path and teacher_name_or_path not in ["self", "disable"]
            else teacher_name_or_path
        )
        if isinstance(teacher, Module):
            SparseAutoModel._log_distillation_teacher_load(teacher, logger)

        return model, teacher

    @staticmethod
    def question_answering_from_pretrained(
        model_name_or_path: str,
        logger: Optional[logging.Logger],
        **kwargs,
    ) -> Module:
        SparseAutoModel._check_tf(model_name_or_path)
        if not kwargs:
            kwargs = {}
        kwargs["from_tf"] = False
        if "state_dict" not in kwargs:
            kwargs["state_dict"] = SparseAutoModel.loadable_state_dict(
                model_name_or_path, logger
            )

        return AutoModelForQuestionAnswering.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

    @staticmethod
    def question_answering_from_pretrained_distil(
        model_name_or_path: str,
        teacher_name_or_path: Optional[str],
        model_kwargs: Dict[str, Any],
        teacher_kwars: Dict[str, Any],
        logger: Optional[logging.Logger],
    ) -> Tuple[Module, Optional[Union[Module, str]]]:
        model = SparseAutoModel.question_answering_from_pretrained(
            model_name_or_path, logger, **model_kwargs
        )
        teacher = (
            SparseAutoModel.question_answering_from_pretrained(
                teacher_name_or_path, logger, **teacher_kwars
            )
            if teacher_name_or_path and teacher_name_or_path not in ["self", "disable"]
            else teacher_name_or_path
        )
        if isinstance(teacher, Module):
            SparseAutoModel._log_distillation_teacher_load(teacher, logger)

        return model, teacher

    @staticmethod
    def sequence_classification_from_pretrained(
        model_name_or_path: str,
        logger: Optional[logging.Logger],
        **kwargs,
    ) -> Module:
        SparseAutoModel._check_tf(model_name_or_path)
        if not kwargs:
            kwargs = {}
        kwargs["from_tf"] = False
        if "state_dict" not in kwargs:
            kwargs["state_dict"] = SparseAutoModel.loadable_state_dict(
                model_name_or_path, logger
            )

        return AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

    @staticmethod
    def sequence_classification_from_pretrained_distil(
        model_name_or_path: str,
        teacher_name_or_path: Optional[str],
        model_kwargs: Dict[str, Any],
        teacher_kwars: Dict[str, Any],
        logger: Optional[logging.Logger],
    ) -> Tuple[Module, Optional[Module]]:
        model = SparseAutoModel.sequence_classification_from_pretrained(
            model_name_or_path, logger, **model_kwargs
        )
        teacher = (
            SparseAutoModel.sequence_classification_from_pretrained(
                teacher_name_or_path, logger, **teacher_kwars
            )
            if teacher_name_or_path and teacher_name_or_path not in ["self", "disable"]
            else teacher_name_or_path
        )
        if isinstance(teacher, Module):
            SparseAutoModel._log_distillation_teacher_load(teacher, logger)

        return model, teacher

    @staticmethod
    def token_classification_from_pretrained(
        model_name_or_path: str,
        logger: Optional[logging.Logger],
        **kwargs,
    ) -> Module:
        SparseAutoModel._check_tf(model_name_or_path)
        if not kwargs:
            kwargs = {}
        kwargs["from_tf"] = False
        if "state_dict" not in kwargs:
            kwargs["state_dict"] = SparseAutoModel.loadable_state_dict(
                model_name_or_path, logger
            )

        return AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

    @staticmethod
    def token_classification_from_pretrained_distil(
        model_name_or_path: str,
        teacher_name_or_path: Optional[str],
        model_kwargs: Dict[str, Any],
        teacher_kwars: Dict[str, Any],
        logger: Optional[logging.Logger],
    ) -> Tuple[Module, Optional[Module]]:
        model = SparseAutoModel.token_classification_from_pretrained(
            model_name_or_path, logger, **model_kwargs
        )
        teacher = (
            SparseAutoModel.token_classification_from_pretrained(
                teacher_name_or_path, logger, **teacher_kwars
            )
            if teacher_name_or_path and teacher_name_or_path not in ["self", "disable"]
            else teacher_name_or_path
        )
        if isinstance(teacher, Module):
            SparseAutoModel._log_distillation_teacher_load(teacher, logger)

        return model, teacher

    @staticmethod
    def loadable_state_dict(
        model_name_or_path: str, logger: Optional[logging.Logger]
    ) -> Optional[Dict[str, Any]]:
        if not model_name_or_path or not os.path.isfile(
            os.path.join(model_name_or_path, WEIGHTS_NAME)
        ):
            return None

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
            return None

        logger.warning(
            "QAT state detected, skipping load of state_dict for model until after "
            f"SparseML recipes have been applied {model_name_or_path}"
        )

        return {}

    @staticmethod
    def _check_tf(model_name_or_path: str):
        if ".ckpt" in model_name_or_path:
            raise ValueError(
                "PyTorch is the only supported model type currently for SparseML "
                "HuggingFace Transformers integration. "
                "Detected a TensorFlow model from model_name_or_path: "
                f"{model_name_or_path}"
            )

    @staticmethod
    def _log_distillation_teacher_load(
        teacher: Module, logger: Optional[logging.Logger]
    ):
        if teacher is None or logger is None:
            return

        teacher_params = filter(lambda p: p.requires_grad, teacher.parameters())
        params = sum(numpy.prod(p.size()) for p in teacher_params)
        logger.info("Loaded distillation teacher with %s parameters", params)
