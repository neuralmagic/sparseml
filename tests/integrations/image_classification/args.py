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
import os
from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, Field

from sparseml.pytorch.image_classification.utils import OPTIMIZERS
from sparseml.pytorch.utils import default_dataset
from sparseml.utils.datasets import default_dataset_path


__all__ = [
    "DEFAULT_SAVE_DIR",
    "ImageClassificationTrainArgs",
    "ImageClassificationExportArgs",
]


DEFAULT_SAVE_DIR = "pytorch_vision_tests/"


class _ImageClassificationBaseArgs(BaseModel):
    # shared args
    dataset: str = Field(
        description="name of dataset to use, imagefolder can be used for custom"
    )
    dataset_path: Union[str, Path] = Field(
        default=default_dataset_path(),
        description=(
            "location of dataset root or where dataset should be downloaded. Defaults "
            "to a cache directory"
        ),
    )
    arch_key: str = Field(
        default=None,
        description=(
            "name of model to create, will be loaded from checkpoint if "
            "none provided"
        ),
    )
    checkpoint_path: Union[str, Path] = Field(
        default=None, description="path to weights checkpoint"
    )
    pretrained: str = Field(defualt="True", description="type of pretrained weights")
    pretrained_dataset: str = Field(default=None, description="checkpoint data name")
    model_kwargs: str = Field(
        default=None, description="json string for model constructor args"
    )
    dataset_kwargs: str = Field(
        default=None, description="json string for dataset constructor args"
    )
    model_tag: str = Field(description="required - tag for model under save_dir")
    save_dir: Union[str, Path] = Field(default=DEFAULT_SAVE_DIR)


class ImageClassificationTrainArgs(_ImageClassificationBaseArgs):
    train_batch_size: int = Field(description="batch size to use in train loop")
    test_batch_size: int = Field(description="batch size to use in eval loop")
    init_lr: float = Field(default=1e-9, description="will be overwritten by recipe")
    recipe_path: Union[str, Path] = Field(
        default=None, description="path to sparsification recipe"
    )
    eval_mode: bool = Field(
        default=True, description="defaults to True to run eval on epoch"
    )
    optim: Literal[tuple(OPTIMIZERS)] = Field(
        default="SGD", description="torch optimizer class to use, default SGD"
    )
    optim_args: str = Field(
        default=json.dumps(
            {
                "momentum": 0.9,
                "nesterov": True,
                "weight_decay": 0.0001,
            }
        ),
        description="json string of arguments to optimizer class",
    )
    logs_dir: Union[str, Path] = Field(
        default=os.path.join(DEFAULT_SAVE_DIR, "tensorboard-logs"),
        description="location to save tensorboard logs",
    )
    save_best_after: int = Field(
        default=1, description="epoch to save best checkpoint after"
    )
    use_mixed_precision: bool = Field(default=False, description="set True for fp16")
    debug_steps: int = Field(
        default=-1, description="number of steps to run per epoch in debug mode"
    )
    device: str = Field(default=default_dataset())
    loader_num_workers: int = Field(default=4, description="num workers per process")
    loader_pin_memory: bool = Field(default=True)
    image_size: int = Field(default=224)
    ffcv: bool = Field(
        default=False,
        description="FFCV if installed will be activated for dataloader acceleration",
    )
    recipe_args: str = Field(
        default=None, description="json string for recipe constructor"
    )


class ImageClassificationExportArgs(_ImageClassificationBaseArgs):
    onnx_opset: int = Field(default=11)
    num_samples: int = Field(
        default=-1, description="number of forward data output samples to produce"
    )
    use_zipfile_serialization_if_available: bool = Field(default=True)
    recipe: Union[str, Path] = Field(
        default=None, description="recipe to apply to model before loading weights"
    )
