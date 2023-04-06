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

import os
import warnings
from argparse import Namespace
from typing import Any, Dict

from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.model import DetectionModel
from ultralytics.yolo.engine.trainer import BaseTrainer


__all__ = ["check_coco128_segmentation", "create_grad_sampler"]


def check_coco128_segmentation(args: Namespace) -> Namespace:
    """
    Checks if the argument 'data' is coco128.yaml and if so,
    replaces it with coco128-seg.yaml.
    :param args: arguments to check
    :return: the updated arguments
    """
    if args.data == "coco128.yaml":
        dataset_name, dataset_extension = os.path.splitext(args.data)
        dataset_yaml = dataset_name + "-seg" + dataset_extension
        warnings.warn(
            f"Dataset yaml {args.data} is not supported for segmentation. "
            f"Attempting to use {dataset_yaml} instead."
        )
        args.data = dataset_yaml
    return args


def create_grad_sampler(
    trainer: BaseTrainer, stride: int, model: DetectionModel
) -> Dict[str, Any]:
    if not hasattr(trainer, "train_loader"):
        # initialize train loader (if not already initialized)
        # and set it as the trainer's attribute
        train_set_path = trainer.trainset
        train_loader, _ = create_dataloader(
            path=train_set_path,
            imgsz=trainer.args.imgsz,
            batch_size=trainer.args.batch,
            stride=stride,
        )
        trainer.train_loader = train_loader

    # convert model's arg to a namespace,
    # this is expected by the trainer's criterion
    model.args = Namespace(**model.args)
    trainer.model = model

    grad_sampler = dict(
        data_loader_builder=trainer._get_data_loader_builder(),
        loss_function=lambda preds, batch: trainer.criterion(preds, batch)[0]
        / train_loader.batch_size,
    )
    return grad_sampler
