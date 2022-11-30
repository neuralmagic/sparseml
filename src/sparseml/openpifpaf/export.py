# Adapted from https://github.com/openpifpaf/openpifpaf

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

import argparse
import logging
import os
from typing import Optional

import onnx
import torch

import openpifpaf
from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter


LOG = logging.getLogger(__name__)


def image_size_warning(basenet_stride, input_w, input_h):
    if input_w % basenet_stride != 1:
        LOG.warning(
            "input width (%d) should be a multiple of basenet "
            "stride (%d) + 1: closest are %d and %d",
            input_w,
            basenet_stride,
            (input_w - 1) // basenet_stride * basenet_stride + 1,
            ((input_w - 1) // basenet_stride + 1) * basenet_stride + 1,
        )

    if input_h % basenet_stride != 1:
        LOG.warning(
            "input height (%d) should be a multiple of basenet "
            "stride (%d) + 1: closest are %d and %d",
            input_h,
            basenet_stride,
            (input_h - 1) // basenet_stride * basenet_stride + 1,
            ((input_h - 1) // basenet_stride + 1) * basenet_stride + 1,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=str,
        default="openpifpaf-onnx-exports",
        help="The path to the directory for saving results",
    )
    parser.add_argument(
        "--name", default="model.onnx", type=str, help="Name of the model file"
    )
    parser.add_argument("--input-width", type=int, default=129)
    parser.add_argument("--input-height", type=int, default=97)
    parser.add_argument(
        "--one-shot", type=str, help="Path to recipe to apply in a zero shot fashion."
    )

    openpifpaf.network.Factory.cli(parser)
    openpifpaf.datasets.cli(parser)
    args = parser.parse_args()

    openpifpaf.network.Factory.configure(args)

    datamodule = openpifpaf.datasets.factory(args.dataset)
    model, _ = openpifpaf.network.Factory().factory(head_metas=datamodule.head_metas)

    export(
        args.save_dir,
        args.name,
        model,
        datamodule,
        args.one_shot,
        args.input_width,
        args.input_height,
    )


def export(
    save_dir: str,
    name: str,
    model,
    datamodule,
    one_shot: Optional[str],
    input_width: int,
    input_height: int,
):
    if one_shot:
        manager = ScheduledModifierManager.from_yaml(one_shot)
        manager.apply(model)

    image_size_warning(model.base_net.stride, input_width, input_height)

    # configure
    openpifpaf.network.heads.CompositeField3.inplace_ops = False
    openpifpaf.network.heads.CompositeField4.inplace_ops = False

    exporter = ModuleExporter(model, save_dir)
    exporter.export_onnx(
        torch.randn(1, 3, input_height, input_width),
        name=name,
        input_names=["input_batch"],
        output_names=[meta.name for meta in datamodule.head_metas],
    )
    onnx.checker.check_model(os.path.join(save_dir, name))
    exporter.create_deployment_folder()


if __name__ == "__main__":
    main()
