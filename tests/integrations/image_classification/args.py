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
from typing import Optional, Union

from pydantic import BaseModel, Field

from sparseml.pytorch.image_classification.utils import OPTIMIZERS
from sparseml.pytorch.utils import default_device
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
        default=None,
        description="name of dataset to use, imagefolder can be used for custom",
    )
    dataset_path: Union[str, Path] = Field(
        default=None,
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
    pretrained: str = Field(default="True", description="type of pretrained weights")
    pretrained_dataset: str = Field(default=None, description="checkpoint data name")
    model_kwargs: str = Field(
        default=None, description="json string for model constructor args"
    )
    dataset_kwargs: str = Field(
        default=None, description="json string for dataset constructor args"
    )
    model_tag: str = Field(description="required - tag for model under save_dir")
    save_dir: Union[str, Path] = Field(default=DEFAULT_SAVE_DIR)

    def __init__(self, **data):
        super().__init__(**data)
        self.__post_init__()

    def __post_init__(self):
        if self.dataset and not self.dataset_path:
            self.dataset_path = default_dataset_path(self.dataset)


class ImageClassificationTrainArgs(_ImageClassificationBaseArgs):
    train_batch_size: int = Field(description="batch size to use in train loop")
    test_batch_size: int = Field(description="batch size to use in eval loop")
    init_lr: float = Field(default=1e-9, description="will be overwritten by recipe")
    gradient_accum_steps: int = Field(
        default=1, description="gradient accumulation steps"
    )
    recipe_path: Union[str, Path] = Field(
        default=None, description="path to sparsification recipe"
    )
    eval_mode: bool = Field(default=False, description="defaults to only run eval")
    optim: str = Field(
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
    device: str = Field(default=default_device())
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
    max_train_steps: int = Field(
        default=-1,
        description="The maximum number of training steps to run per epoch. If "
        "negative, will run for the entire training set",
    )
    max_eval_steps: int = Field(
        default=-1,
        description="The maximum number of eval steps to run per epoch. If negative, "
        "will run for the entire validation set",
    )
    one_shot: bool = Field(
        default=False,
        description="Apply recipe in a one-shot fashion and save the model",
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.__post_init__()

    def __post_init__(self):
        if self.optim not in tuple(OPTIMIZERS):
            raise ValueError(
                f"keyword --optim must be one of {tuple(OPTIMIZERS)}. "
                f"Instead, received: {self.optim}"
            )


class ImageClassificationExportArgs(_ImageClassificationBaseArgs):
    model_tag: Optional[str] = Field(
        Default=None, description="required - tag for model under save_dir"
    )
    onnx_opset: int = Field(
        default=13, description="The onnx opset to use for exporting the model"
    )
    num_samples: int = Field(
        default=-1, description="number of forward data output samples to produce"
    )
    use_zipfile_serialization_if_available: bool = Field(default=True)
    recipe: Union[str, Path] = Field(
        default=None, description="recipe to apply to model before loading weights"
    )
    num_classes: Optional[int] = Field(
        default=None, description="number of classes for model load/export"
    )


class ImageClassificationDeployArgs(BaseModel):
    model_path: Optional[str] = Field(
        default=None,
        description=("Path to directory where model onnx file is stored"),
    )
