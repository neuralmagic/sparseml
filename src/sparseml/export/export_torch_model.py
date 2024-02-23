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
from pathlib import Path
from typing import Union

import onnx
import torch

from sparseml.exporters import ExportTargets
from sparseml.exporters.onnx_to_deepsparse import ONNXToDeepsparse
from sparseml.pytorch.opset import TORCH_DEFAULT_ONNX_OPSET
from sparseml.pytorch.torch_to_onnx_exporter import TorchToONNX


__all__ = ["export_model"]


def export_model(
    model: torch.nn.Module,
    sample_data: torch.Tensor,
    target_path: Union[Path, str],
    onnx_model_name: str,
    deployment_target: str = "deepsparse",
    opset: int = TORCH_DEFAULT_ONNX_OPSET,
    **kwargs,
) -> str:
    """
    Exports the torch model to the deployment target

    :param model: The torch model to export
    :param sample_data: The sample data to use for the export
    :param target_path: The path to export the model to
    :param onnx_model_name: The name to save  the exported ONNX model as
    :param deployment_target: The deployment target to export to. Defaults to deepsparse
    :param opset: The opset to use for the export. Defaults to TORCH_DEFAULT_ONNX_OPSET
    :param kwargs: Additional kwargs to pass to the TorchToONNX exporter
    :return: The path to the exported model
    """

    model.eval()
    path_to_exported_model = os.path.join(target_path, onnx_model_name)
    exporter = TorchToONNX(sample_batch=sample_data, opset=opset, **kwargs)

    # If performing deepsparse transforms, don't split the initial onnx export
    do_deploy_deepsparse = deployment_target == ExportTargets.deepsparse.value
    exporter.export(
        model, path_to_exported_model, do_split_external_data=(not do_deploy_deepsparse)
    )
    if do_deploy_deepsparse:
        exporter = ONNXToDeepsparse()
        model = onnx.load(path_to_exported_model)
        exporter.export(model, path_to_exported_model, do_split_external_data=True)
        return path_to_exported_model
    if deployment_target == ExportTargets.onnx.value:
        return path_to_exported_model
    else:
        raise ValueError(f"Unsupported deployment target: {deployment_target}")
