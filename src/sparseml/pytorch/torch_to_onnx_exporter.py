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

import collections
import logging
import tempfile
import warnings
from copy import deepcopy
from typing import Any, Dict, Iterable, List

import onnx
import torch
from packaging import version

from sparseml.exporters import transforms as sparseml_transforms
from sparseml.exporters.base_exporter import BaseExporter
from sparseml.exporters.transforms.base_transform import BaseTransform
from sparseml.pytorch import _PARSED_TORCH_VERSION
from sparseml.pytorch.opset import TORCH_DEFAULT_ONNX_OPSET
from sparseml.pytorch.utils.helpers import tensors_module_forward, tensors_to_device
from sparseml.pytorch.utils.model import is_parallel_model


_LOGGER = logging.getLogger(__name__)


class TorchToONNX(BaseExporter):
    """
    Transforms a `torch.nn.Module` into an `onnx.ModelProto` using `torch.onnx.export`.

    Example usage:

    ```python
    model = torchvision.models.resnet18().eval()
    exporter = TorchToONNX(sample_batch=torch.randn(1, 3, 224, 224))
    exporter.export(model, "resnest18.onnx")
    ```

    :param sample_batch: the batch to export an onnx for, handles creating the
        static graph for onnx as well as setting dimensions
    :param opset: onnx opset to use for exported model.
        Default is based on torch version.
    :param disable_bn_fusing: torch >= 1.7.0 only. Set True to disable batch norm
        fusing during torch export. Default and suggested setting is True. Batch
        norm fusing will change the exported parameter names as well as affect
        sensitivity analyses of the exported graph.  Additionally, the DeepSparse
        inference engine, and other engines, perform batch norm fusing at model
        compilation.
    :param export_kwargs: kwargs to be passed as is to the torch.onnx.export api
        call. Useful to pass in dyanmic_axes, input_names, output_names, etc.
        See more on the torch.onnx.export api spec in the PyTorch docs:
        https://pytorch.org/docs/stable/onnx.html
    """

    def __init__(
        self,
        sample_batch: Any,
        opset: int = TORCH_DEFAULT_ONNX_OPSET,
        disable_bn_fusing: bool = True,
        **export_kwargs,
    ):
        super().__init__(
            [
                # NOTE: this first transform is what transforms
                # the torch.nn.Module into an onnx.ModelProto
                _TorchOnnxExport(
                    sample_batch, opset, disable_bn_fusing, **export_kwargs
                ),
                # NOTE: the remainder of these act on onnx.ModelProto
                sparseml_transforms.FoldIdentityInitializers(),
                sparseml_transforms.FlattenQParams(),
                sparseml_transforms.UnwrapBatchNorms(),
                sparseml_transforms.DeleteTrivialOnnxAdds(),
            ],
        )

    def pre_validate(self, module: torch.nn.Module) -> torch.nn.Module:
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, found {type(module)}")

        if is_parallel_model(module):
            module = module.module

        return deepcopy(module).to("cpu").eval()

    def post_validate(self, model: onnx.ModelProto) -> onnx.ModelProto:
        if not isinstance(model, onnx.ModelProto):
            raise TypeError(f"Expected onnx.ModelProto, found {type(model)}")
        return model

    def export(self, pre_transforms_model: torch.nn.Module, file_path: str):
        post_transforms_model: onnx.ModelProto = self.apply(pre_transforms_model)
        onnx.save(post_transforms_model, file_path)


class _TorchOnnxExport(BaseTransform):
    def __init__(
        self,
        sample_batch: Any,
        opset: int = TORCH_DEFAULT_ONNX_OPSET,
        disable_bn_fusing: bool = True,
        **export_kwargs,
    ):
        super().__init__()

        if _PARSED_TORCH_VERSION >= version.parse("1.10.0") and opset < 13:
            warnings.warn(
                "Exporting onnx with QAT and opset < 13 may result in errors. "
                "Please use opset>=13 with QAT. "
                "See https://github.com/pytorch/pytorch/issues/77455 for more info. "
            )

        if isinstance(sample_batch, Dict) and not isinstance(
            sample_batch, collections.OrderedDict
        ):
            warnings.warn(
                "Sample inputs passed into the ONNX exporter should be in "
                "the same order defined in the model forward function. "
                "Consider using OrderedDict for this purpose.",
                UserWarning,
            )

        self.sample_batch = sample_batch
        self.opset = opset
        self.disable_bn_fusing = disable_bn_fusing
        self.export_kwargs = export_kwargs or {}

    def pre_validate(self, model: torch.nn.Module) -> torch.nn.Module:
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Expected torch.nn.Module, found {type(model)}")
        return model

    def post_validate(self, model: onnx.ModelProto) -> onnx.ModelProto:
        if not isinstance(model, onnx.ModelProto):
            raise TypeError(f"Expected onnx.ModelProto, found {type(model)}")
        return model

    def transform(self, module: torch.nn.Module) -> onnx.ModelProto:
        tmp = tempfile.NamedTemporaryFile("w")
        file_path = tmp.name

        _LOGGER.debug(f"Saving onnx model to {file_path}")

        export_kwargs = deepcopy(self.export_kwargs)

        sample_batch = tensors_to_device(self.sample_batch, "cpu")

        module = deepcopy(module).cpu()

        with torch.no_grad():
            out = tensors_module_forward(sample_batch, module, check_feat_lab_inp=False)

        if "input_names" not in export_kwargs:
            if isinstance(sample_batch, torch.Tensor):
                export_kwargs["input_names"] = ["input"]
            elif isinstance(sample_batch, Dict):
                export_kwargs["input_names"] = list(sample_batch.keys())
                sample_batch = tuple(
                    [sample_batch[f] for f in export_kwargs["input_names"]]
                )
            elif isinstance(sample_batch, Iterable):
                export_kwargs["input_names"] = [
                    "input_{}".format(index)
                    for index, _ in enumerate(iter(sample_batch))
                ]
                if isinstance(sample_batch, List):
                    # torch.onnx.export requires tuple
                    sample_batch = tuple(sample_batch)

        if "output_names" not in export_kwargs:
            export_kwargs["output_names"] = _get_output_names(out)

        dynamic_axes = {
            tensor_name: {0: "batch"}
            for tensor_name in (
                export_kwargs["input_names"] + export_kwargs["output_names"]
            )
        }

        # disable active quantization observers because they cannot be exported
        disabled_observers = []
        for submodule in module.modules():
            if (
                hasattr(submodule, "observer_enabled")
                and submodule.observer_enabled[0] == 1
            ):
                submodule.observer_enabled[0] = 0
                disabled_observers.append(submodule)

        is_quant_module = any(
            hasattr(submodule, "qconfig") and submodule.qconfig
            for submodule in module.modules()
        )
        if (
            _PARSED_TORCH_VERSION >= version.parse("1.7")
            and not is_quant_module
            and self.disable_bn_fusing
        ):
            # prevent batch norm fusing by adding a trivial operation before every
            # batch norm layer
            _LOGGER.debug("Wrapping batch norms")
            _wrap_batch_norms(module)

        kwargs = dict(
            model=module,
            args=sample_batch,
            f=file_path,
            verbose=False,
            opset_version=self.opset,
            dynamic_axes=dynamic_axes,
            **export_kwargs,
        )

        if _PARSED_TORCH_VERSION < version.parse("1.10.0"):
            kwargs["strip_doc_string"] = True
        else:
            kwargs["training"] = torch.onnx.TrainingMode.PRESERVE
            kwargs["do_constant_folding"] = not module.training
            kwargs["keep_initializers_as_inputs"] = False

        _LOGGER.debug(f"Running torch.onnx.export with {kwargs}")
        torch.onnx.export(**kwargs)

        # re-enable disabled quantization observers
        for submodule in disabled_observers:
            submodule.observer_enabled[0] = 1

        return onnx.load(file_path)


def _get_output_names(out: Any):
    """
    Get name of output tensors

    :param out: outputs of the model
    :return: list of names
    """
    output_names = None
    if isinstance(out, torch.Tensor):
        output_names = ["output"]
    elif hasattr(out, "keys") and callable(out.keys):
        output_names = list(out.keys())
    elif isinstance(out, Iterable):
        output_names = ["output_{}".format(index) for index, _ in enumerate(iter(out))]
    return output_names


def _wrap_batch_norms(module: torch.nn.Module) -> bool:
    # wrap all batch norm layers in module with a trivial wrapper
    # to prevent BN fusing during export
    batch_norms_wrapped = False
    for name, submodule in module.named_modules():
        if (
            isinstance(submodule, torch.nn.BatchNorm1d)
            or isinstance(submodule, torch.nn.BatchNorm2d)
            or isinstance(submodule, torch.nn.BatchNorm3d)
        ):
            submodule_path = name.split(".")
            parent_module = _get_submodule(module, submodule_path[:-1])
            setattr(parent_module, submodule_path[-1], _AddNoOpWrapper(submodule))
            batch_norms_wrapped = True
    return batch_norms_wrapped


def _get_submodule(module: torch.nn.Module, path: List[str]) -> torch.nn.Module:
    if not path:
        return module
    return _get_submodule(getattr(module, path[0]), path[1:])


class _AddNoOpWrapper(torch.nn.Module):
    # trivial wrapper to break-up Conv-BN blocks

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.bn_wrapper_replace_me = module

    def forward(self, inp):
        inp = inp + 0  # no-op
        return self.bn_wrapper_replace_me(inp)
