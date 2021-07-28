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

"""
Export PyTorch models to the local device
"""
import collections
import logging
import os
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy
import onnx
import torch
from onnx import numpy_helper
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from sparseml.pytorch.utils.helpers import (
    tensors_export,
    tensors_module_forward,
    tensors_to_device,
)
from sparseml.pytorch.utils.model import (
    is_parallel_model,
    save_model,
    script_model,
    trace_model,
)
from sparseml.pytorch.utils.quantization import quantize_torch_qat_export
from sparseml.utils import clean_path, create_parent_dirs


__all__ = ["ModuleExporter"]


DEFAULT_ONNX_OPSET = 9 if torch.__version__ < "1.3" else 11
_LOGGER = logging.getLogger(__name__)


class ModuleExporter(object):
    """
    An exporter for exporting PyTorch modules into ONNX format
    as well as numpy arrays for the input and output tensors.

    :param module: the module to export
    :param output_dir: the directory to export the module and extras to
    """

    def __init__(
        self,
        module: Module,
        output_dir: str,
    ):
        if is_parallel_model(module):
            module = module.module

        self._module = deepcopy(module).to("cpu").eval()
        self._output_dir = clean_path(output_dir)

    def export_to_zoo(
        self,
        dataloader: DataLoader,
        original_dataloader: Optional[DataLoader] = None,
        shuffle: bool = False,
        max_samples: int = 20,
        data_split_cb: Optional[Callable[[Any], Tuple[Any, Any]]] = None,
        label_mapping_cb: Optional[Callable[[Any], Any]] = None,
        trace_script: bool = False,
        fail_on_torchscript_failure: bool = True,
        export_entire_model: bool = False,
    ):
        """
        Creates and exports all related content of module including
        sample data, onnx, pytorch and torchscript.

        :param dataloader: DataLoader used to generate sample data
        :param original_dataloader: Optional dataloader to obtain the untransformed
            image.
        :param shuffle: Whether to shuffle sample data
        :param max_samples: Max number of sample data to create
        :param data_split_cb: Optional callback function to split data sample into
            a tuple (features,labels). If not provided will assume dataloader
            returns a tuple (features,labels).
        :param label_mapping_cb: Optional callback function to mapping dataset label to
            other   formats.
        :param dataset_wrapper: Wrapper function for the dataset to add original data
            to each sample. If set to None will default to use the
            'iter_dataset_with_orig_wrapper' function.
        :param trace_script: If true, creates torchscript via tracing. Otherwise,
            creates the torchscripe via scripting.
        :param fail_on_torchscript_failure: If true, fails if torchscript is unable
            to export model.
        :param export_entire_model: Exports entire file instead of state_dict
        """
        sample_batches = []
        sample_labels = []
        sample_originals = None
        if original_dataloader is not None:
            sample_originals = []
            for originals in original_dataloader:
                sample_originals.append(originals)
                if len(sample_originals) == max_samples:
                    break

        for sample in dataloader:
            if data_split_cb is not None:
                features, labels = data_split_cb(sample)
            else:
                features, labels = sample
            if label_mapping_cb:
                labels = label_mapping_cb(labels)

            sample_batches.append(features)
            sample_labels.append(labels)
            if len(sample_batches) == max_samples:
                break

        self.export_onnx(sample_batch=sample_batches[0])
        self.export_pytorch(export_entire_model=export_entire_model)
        try:
            if trace_script:
                self.export_torchscript(sample_batch=sample_batches[0])
            else:
                self.export_torchscript()
        except Exception as e:
            if fail_on_torchscript_failure:
                raise e
            else:
                _LOGGER.warn(
                    f"Unable to create torchscript file. Following error occurred: {e}"
                )

        self.export_samples(
            sample_batches,
            sample_labels=sample_labels,
            sample_originals=sample_originals,
        )

    @classmethod
    def get_output_names(cls, out: Any):
        """
        Get name of output tensors

        :param out: outputs of the model
        :return: list of names
        """
        output_names = None
        if isinstance(out, Tensor):
            output_names = ["output"]
        elif isinstance(out, Iterable):
            output_names = [
                "output_{}".format(index) for index, _ in enumerate(iter(out))
            ]
        return output_names

    def export_onnx(
        self,
        sample_batch: Any,
        name: str = "model.onnx",
        opset: int = DEFAULT_ONNX_OPSET,
        disable_bn_fusing: bool = True,
        convert_qat: bool = False,
        **export_kwargs,
    ):
        """
        Export an onnx file for the current module and for a sample batch.
        Sample batch used to feed through the model to freeze the graph for a
        particular execution.

        :param sample_batch: the batch to export an onnx for, handles creating the
            static graph for onnx as well as setting dimensions
        :param name: name of the onnx file to save
        :param opset: onnx opset to use for exported model. Default is 11, if torch
            version is 1.2 or below, default is 9
        :param disable_bn_fusing: torch >= 1.7.0 only. Set True to disable batch norm
            fusing during torch export. Default and suggested setting is True. Batch
            norm fusing will change the exported parameter names as well as affect
            sensitivity analyses of the exported graph.  Additionally, the DeepSparse
            inference engine, and other engines, perform batch norm fusing at model
            compilation.
        :param convert_qat: if True and quantization aware training is detected in
            the module being exported, the resulting QAT ONNX model will be converted
            to a fully quantized ONNX model using `quantize_torch_qat_export`. Default
            is False.
        :param export_kwargs: kwargs to be passed as is to the torch.onnx.export api
            call. Useful to pass in dyanmic_axes, input_names, output_names, etc.
            See more on the torch.onnx.export api spec in the PyTorch docs:
            https://pytorch.org/docs/stable/onnx.html
        """
        if not export_kwargs:
            export_kwargs = {}

        if isinstance(sample_batch, Dict) and not isinstance(
            sample_batch, collections.OrderedDict
        ):
            warnings.warn(
                "Sample inputs passed into the ONNX exporter should be in "
                "the same order defined in the model forward function. "
                "Consider using OrderedDict for this purpose.",
                UserWarning,
            )

        sample_batch = tensors_to_device(sample_batch, "cpu")
        onnx_path = os.path.join(self._output_dir, name)
        create_parent_dirs(onnx_path)

        with torch.no_grad():
            out = tensors_module_forward(
                sample_batch, self._module, check_feat_lab_inp=False
            )

        if "input_names" not in export_kwargs:
            if isinstance(sample_batch, Tensor):
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
                    sample_batch = tuple(
                        sample_batch
                    )  # torch.onnx.export requires tuple

        if "output_names" not in export_kwargs:
            export_kwargs["output_names"] = self.get_output_names(out)

        # disable active quantization observers because they cannot be exported
        disabled_observers = []
        for submodule in self._module.modules():
            if (
                hasattr(submodule, "observer_enabled")
                and submodule.observer_enabled[0] == 1
            ):
                submodule.observer_enabled[0] = 0
                disabled_observers.append(submodule)

        is_quant_module = any(
            hasattr(submodule, "qconfig") and submodule.qconfig
            for submodule in self._module.modules()
        )
        batch_norms_wrapped = False
        if torch.__version__ >= "1.7" and not is_quant_module and disable_bn_fusing:
            # prevent batch norm fusing by adding a trivial operation before every
            # batch norm layer
            export_module = deepcopy(self._module)
            batch_norms_wrapped = _wrap_batch_norms(export_module)
        else:
            export_module = self._module

        torch.onnx.export(
            export_module,
            sample_batch,
            onnx_path,
            strip_doc_string=True,
            verbose=False,
            opset_version=opset,
            **export_kwargs,
        )

        # re-enable disabled quantization observers
        for submodule in disabled_observers:
            submodule.observer_enabled[0] = 1

        # clean up graph from any injected / wrapped operations
        if batch_norms_wrapped:
            onnx_model = onnx.load(onnx_path)
            _delete_trivial_onnx_adds(onnx_model)
            onnx.save(onnx_model, onnx_path)

        if convert_qat and is_quant_module:
            # overwrite exported model with fully quantized version
            quantize_torch_qat_export(model=onnx_path, output_file_path=onnx_path)

    def export_torchscript(
        self,
        name: str = "model.pts",
        sample_batch: Optional[Any] = None,
    ):
        """
        Export the torchscript into a pts file within a framework directory. If
        a sample batch is provided, will create torchscript model in trace mode.
        Otherwise uses script to create torchscript.

        :param name: name of the torchscript file to save
        :param sample_batch: If provided, will create torchscript model via tracing
            using the sample_batch
        """
        path = os.path.join(self._output_dir, "framework", name)
        create_parent_dirs(path)
        if sample_batch:
            trace_model(path, self._module, sample_batch)
        else:
            script_model(path, self._module)

    def export_pytorch(
        self,
        optimizer: Optimizer = None,
        epoch: int = None,
        name: str = "model.pth",
        use_zipfile_serialization_if_available: bool = True,
        include_modifiers: bool = False,
        export_entire_model: bool = False,
    ):
        """
        Export the pytorch state dicts into pth file within a
        pytorch framework directory.

        :param optimizer: optional optimizer to export along with the module
        :param epoch: optional epoch to export along with the module
        :param name: name of the pytorch file to save
        :param use_zipfile_serialization_if_available: for torch >= 1.6.0 only
            exports the Module's state dict using the new zipfile serialization
        :param include_modifiers: if True, and a ScheduledOptimizer is provided
            as the optimizer, the associated ScheduledModifierManager and its
            Modifiers will be exported under the 'manager' key. Default is False
        :param export_entire_model: Exports entire file instead of state_dict
        """
        pytorch_path = os.path.join(self._output_dir, "framework")
        pth_path = os.path.join(pytorch_path, name)
        create_parent_dirs(pth_path)

        if export_entire_model:
            torch.save(self._module, pth_path)
        else:
            save_model(
                pth_path,
                self._module,
                optimizer,
                epoch,
                use_zipfile_serialization_if_available=(
                    use_zipfile_serialization_if_available
                ),
                include_modifiers=include_modifiers,
            )

    def export_samples(
        self,
        sample_batches: List[Any],
        sample_labels: Optional[List[Any]] = None,
        sample_originals: Optional[List[Any]] = None,
        exp_counter: int = 0,
    ):
        """
        Export a set list of sample batches as inputs and outputs through the model.

        :param sample_batches: a list of the sample batches to feed through the module
                               for saving inputs and outputs
        :param sample_labels: an optional list of sample labels that correspond to the
            the batches for saving
        :param exp_counter: the counter to start exporting the tensor files at
        """
        sample_batches = [tensors_to_device(batch, "cpu") for batch in sample_batches]
        inputs_dir = os.path.join(self._output_dir, "sample-inputs")
        outputs_dir = os.path.join(self._output_dir, "sample-outputs")
        labels_dir = os.path.join(self._output_dir, "sample-labels")
        originals_dir = os.path.join(self._output_dir, "sample-originals")

        with torch.no_grad():
            for batch, lab, orig in zip(
                sample_batches,
                sample_labels if sample_labels else [None for _ in sample_batches],
                sample_originals
                if sample_originals
                else [None for _ in sample_batches],
            ):
                out = tensors_module_forward(batch, self._module)

                exported_input = tensors_export(
                    batch,
                    inputs_dir,
                    name_prefix="inp",
                    counter=exp_counter,
                    break_batch=True,
                )
                if isinstance(out, dict):
                    new_out = []
                    for key in out:
                        new_out.append(out[key])
                    out = new_out
                exported_output = tensors_export(
                    out,
                    outputs_dir,
                    name_prefix="out",
                    counter=exp_counter,
                    break_batch=True,
                )

                if lab is not None:
                    tensors_export(
                        lab, labels_dir, "lab", counter=exp_counter, break_batch=True
                    )

                if orig is not None:
                    tensors_export(
                        orig,
                        originals_dir,
                        "orig",
                        counter=exp_counter,
                        break_batch=True,
                    )

                assert len(exported_input) == len(exported_output)
                exp_counter += len(exported_input)


class _AddNoOpWrapper(Module):
    # trivial wrapper to break-up Conv-BN blocks

    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, inp):
        inp = inp + 0  # no-op
        return self.module(inp)


def _get_submodule(module: Module, path: List[str]) -> Module:
    if not path:
        return module
    return _get_submodule(getattr(module, path[0]), path[1:])


def _wrap_batch_norms(module: Module) -> bool:
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


def _delete_trivial_onnx_adds(model: onnx.ModelProto):
    # delete all add nodes in the graph with second inputs as constant nodes set to 0
    add_nodes = [node for node in model.graph.node if node.op_type == "Add"]
    for add_node in add_nodes:
        try:
            add_const_node = [
                node for node in model.graph.node if node.output[0] == add_node.input[1]
            ][0]
            add_const_val = numpy_helper.to_array(add_const_node.attribute[0].t)
            if numpy.all(add_const_val == 0.0):
                # update graph edges
                parent_node = [
                    node
                    for node in model.graph.node
                    if add_node.input[0] in node.output
                ]
                if not parent_node:
                    continue
                parent_node[0].output[0] = add_node.output[0]
                # remove node and constant
                model.graph.node.remove(add_node)
                model.graph.node.remove(add_const_node)
        except Exception:  # skip node on any error
            continue
