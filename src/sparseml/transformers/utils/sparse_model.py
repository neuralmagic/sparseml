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
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.nn import Module
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PreTrainedModel,
)
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME

from sparseml.pytorch.model_load.helpers import (
    apply_recipe_structure_to_model,
    log_model_load,
)
from sparseml.transformers.compression import CompressionConfig, ModelCompressor
from sparseml.transformers.utils.helpers import resolve_recipe
from sparseml.utils import download_zoo_training_dir
from sparseml.utils.fsdp.context import main_process_first_context


__all__ = ["SparseAutoModel", "SparseAutoModelForCausalLM", "get_shared_tokenizer_src"]


_LOGGER = logging.getLogger(__name__)

SPARSITY_CONFIG_NAME = "sparsity_config"


class SparseAutoModelForCausalLM(AutoModelForCausalLM):
    """
    SparseML wrapper for the AutoModelForCausalLM class
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        recipe: Optional[Union[str, Path]] = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        """
        A wrapper around the AutoModelForCausalLM.from_pretrained method that
        enables the loading of a SparseML recipe file to apply to the model

        :param pretrained_model_name_or_path: the name of or path to the model to load
        :param recipe: the path to the recipe file to apply to the model. Can be a
            string or Path object. If None, a recipe will be searched for in the
            pretrained_model_name_or_path directory and applied if found
        :return the created model for causal language modeling
        """

        def skip(*args, **kwargs):
            pass

        # Skip the initializer step. This accelerates the loading
        # of the models, especially for the quantized models
        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        pretrained_model_name_or_path = (
            pretrained_model_name_or_path.as_posix()
            if isinstance(pretrained_model_name_or_path, Path)
            else pretrained_model_name_or_path
        )

        if pretrained_model_name_or_path.startswith("zoo:"):
            _LOGGER.debug(
                "Passed zoo stub to SparseAutoModelForCausalLM object. "
                "Loading model from SparseZoo training files..."
            )
            with main_process_first_context():
                pretrained_model_name_or_path = download_zoo_training_dir(
                    zoo_stub=pretrained_model_name_or_path
                )

        # TODO: should work for huggingface too
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        sparsity_config = getattr(config, SPARSITY_CONFIG_NAME, None)
        if sparsity_config is not None:
            # need to uncompress the model
            format = sparsity_config.get("format", "dense")
            sparsity_config = CompressionConfig.load_from_registry(
                format, **sparsity_config
            )
            compressor = ModelCompressor.load_from_registry(
                format, config=sparsity_config
            )

        # temporarily set the log level to error, to ignore printing out long missing
        # and unexpected key error messages (these are EXPECTED for quantized models)
        logger = logging.getLogger("transformers.modeling_utils")
        restore_log_level = logger.getEffectiveLevel()
        logger.setLevel(level=logging.ERROR)
        model = super(AutoModelForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        logger.setLevel(level=restore_log_level)

        # If model is compressed on disk, decompress and load the weights
        if sparsity_config is not None:
            dense_gen = compressor.decompress(pretrained_model_name_or_path)
            for name, data in tqdm(dense_gen, desc="Decompressing model"):
                ModelCompressor.replace_layer(name, data, model)
        setattr(model, SPARSITY_CONFIG_NAME, sparsity_config)

        recipe = resolve_recipe(recipe, pretrained_model_name_or_path)
        recipe = "zoo:llama2-7b-open_platypus_orca_llama2_pretrain-pruned60"
        if recipe:
            apply_recipe_structure_to_model(
                model=model,
                model_path=pretrained_model_name_or_path,
                recipe_path=recipe,
            )
        return model

    @staticmethod
    def save_pretrained(
        model: PreTrainedModel,
        save_directory: str,
        sparsity_config: Optional[CompressionConfig] = None,
        save_dense: bool = False,
        **kwargs,
    ):
        if save_dense:
            return model.save_pretrained(save_directory, **kwargs)

        if sparsity_config is None:
            # attempt to infer sparsity config if not provided
            sparsity_config = getattr(model, SPARSITY_CONFIG_NAME, None)
            if sparsity_config is None:
                # try to infer a config from the model
                sparsity_config = SparseAutoModelForCausalLM.infer_compression_config(
                    model
                )

        if sparsity_config is None:
            # no config found, save as dense
            return model.save_pretrained(save_directory, **kwargs)

        # if we've gotten to this point we can run compression since we have a config

        kwargs["safe_serialization"] = True
        compressor = ModelCompressor.load_from_registry(
            sparsity_config.format, config=sparsity_config
        )

        compressed_state_dict = compressor.compress(model.state_dict())
        kwargs["state_dict"] = compressed_state_dict

        model.save_pretrained(save_directory, **kwargs)
        sparsity_config_data = sparsity_config.dict()
        config_file_path = os.path.join(save_directory, CONFIG_NAME)

        with open(config_file_path, "r") as config_file:
            config_data = json.load(config_file)
        config_data[SPARSITY_CONFIG_NAME] = sparsity_config_data
        with open(config_file_path, "w") as config_file:
            json.dump(config_data, config_file, indent=4, sort_keys=True)

    @staticmethod
    def save_compressed(
        model: PreTrainedModel,
        save_directory: str,
        sparsity_config: Optional[CompressionConfig] = None,
        **kwargs,
    ):
        if sparsity_config is None:
            # attempt to infer sparsity config if not provided
            if hasattr(model, SPARSITY_CONFIG_NAME):
                sparsity_config = getattr(model, SPARSITY_CONFIG_NAME)
                if sparsity_config is None:
                    # force dense compression config if can't infer
                    sparsity_config = CompressionConfig.load_from_registry("dense")

        return SparseAutoModelForCausalLM.save_pretrained(
            model=model,
            save_directory=save_directory,
            sparsity_config=sparsity_config,
            **kwargs,
        )

    @staticmethod
    def infer_compression_config(model: Module):
        from sparseml.pytorch.utils import ModuleSparsificationInfo

        info = ModuleSparsificationInfo(model)
        global_sparsity = info.params_sparse_percent
        if global_sparsity < 0.05:
            return None
        import sparseml.core.session as session_manager

        current_session = session_manager.active_session()
        stage_modifiers = current_session.lifecycle.modifiers
        sparsity_structure = "unstructured"
        for stage in stage_modifiers:
            if stage.applied:
                for modifier in stage.modifiers:
                    if hasattr(modifier, "mask_structure"):
                        sparsity_structure = modifier.mask_structure
                        break
        return CompressionConfig.load_from_registry(
            "sparse_bitmask",
            global_sparsity=global_sparsity,
            sparsity_structure=sparsity_structure,
        )


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

        log_model_load(model, model_name_or_path, model_type, delayed)

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
        log_model_load(model, model_name_or_path, model_type, delayed)

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
        model_type: str = "model",
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
        log_model_load(model, model_name_or_path, model_type, delayed)

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
    def text_generation_from_pretrained(
        model_name_or_path: str,
        sequence_length: Optional[int] = None,
        recipe: Optional[Union[str, Path]] = None,
        trust_remote_code: bool = False,
        torch_dtype: Union[str, torch.dtype] = "auto",
        **kwargs,
    ) -> Module:
        """
        :param model_name_or_path: the name of or path to the model to load
        :param sequence_length: the maximum length of the sequence to generate.
            If None, will use the default sequence length for the model.
            Defaults to None.
        :param recipe: the recipe to apply to the model. If None, no recipe is applied
        :param trust_remote_code: related to trust_remote_code in HF transformers.
            If True, will execute the modelling code from the model directory
            (if present). Defaults to False.
        :param torch_dtype: the torch dtype to use for the model. If "auto", will
            use the default dtype for the model. Defaults to "auto".
        :return: the created model for text generation
        """

        model = SparseAutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            recipe=recipe,
            **kwargs,
        )
        if sequence_length is not None:
            model.seqlen = sequence_length

        return model

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
        log_model_load(model, model_name_or_path, model_type, delayed)

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

        return None, True

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
