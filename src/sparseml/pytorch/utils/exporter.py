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
import json
import logging
import os
import shutil
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy
import onnx
import torch
from onnx import numpy_helper
from packaging import version
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from sparseml.onnx.utils import ONNXGraph
from sparseml.pytorch.opset import TORCH_DEFAULT_ONNX_OPSET
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
from sparseml.utils import clean_path, create_parent_dirs


__all__ = [
    "ModuleExporter",
    "export_onnx",
]

_PARSED_TORCH_VERSION = version.parse(torch.__version__)

MODEL_ONNX_NAME = "model.onnx"
CONFIG_JSON_NAME = "config.json"
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
        Get name of output tensors. Derived exporters specific to frameworks
        could override this method
        :param out: outputs of the model
        :return: list of names
        """
        return _get_output_names(out)

    def export_onnx(
        self,
        sample_batch: Any,
        name: str = MODEL_ONNX_NAME,
        opset: int = TORCH_DEFAULT_ONNX_OPSET,
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
        :param opset: onnx opset to use for exported model.
            Default is based on torch version.
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
        if "output_names" not in export_kwargs:
            sample_batch = tensors_to_device(sample_batch, "cpu")
            module = deepcopy(self._module).cpu()
            module.eval()
            with torch.no_grad():
                out = tensors_module_forward(
                    sample_batch, module, check_feat_lab_inp=False
                )
                export_kwargs["output_names"] = self.get_output_names(out)
        export_onnx(
            module=self._module,
            sample_batch=sample_batch,
            file_path=os.path.join(self._output_dir, name),
            opset=opset,
            disable_bn_fusing=disable_bn_fusing,
            convert_qat=convert_qat,
            **export_kwargs,
        )

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

    def create_deployment_folder(
        self, labels_to_class_mapping: Optional[Union[str, Dict[int, str]]] = None
    ):
        """
        Create a deployment folder inside the `self._output_dir` directory.

        :param labels_to_class_mapping: information about the mapping
            from integer labels to string class names.
            Can be either a string (path to the .json serialized dictionary)
            or a dictionary. Default is None
        """
        deployment_folder_dir = os.path.join(self._output_dir, "deployment")

        if os.path.isdir(deployment_folder_dir):
            shutil.rmtree(deployment_folder_dir)
        os.makedirs(deployment_folder_dir)
        _LOGGER.info(f"Created deployment folder at {deployment_folder_dir}")

        # copy over model onnx
        expected_onnx_model_dir = os.path.join(self._output_dir, MODEL_ONNX_NAME)
        deployment_onnx_model_dir = os.path.join(deployment_folder_dir, MODEL_ONNX_NAME)
        _copy_file(src=expected_onnx_model_dir, target=deployment_onnx_model_dir)
        _LOGGER.info(
            f"Saved {MODEL_ONNX_NAME} in the deployment "
            f"folder at {deployment_onnx_model_dir}"
        )

        # create config.json
        config_file_path = _create_config_file(save_dir=deployment_folder_dir)

        if labels_to_class_mapping:
            # append `labels_to_class_mapping` info to config.json
            _save_label_to_class_mapping(
                labels_to_class_mapping=labels_to_class_mapping,
                config_file_path=config_file_path,
            )

    def export_pytorch(
        self,
        optimizer: Optional[Optimizer] = None,
        recipe: Optional[str] = None,
        epoch: Optional[int] = None,
        name: str = "model.pth",
        use_zipfile_serialization_if_available: bool = True,
        include_modifiers: bool = False,
        export_entire_model: bool = False,
        arch_key: Optional[str] = None,
    ):
        """
        Export the pytorch state dicts into pth file within a
        pytorch framework directory.

        :param optimizer: optional optimizer to export along with the module
        :param recipe: the recipe used to obtain the model
        :param epoch: optional epoch to export along with the module
        :param name: name of the pytorch file to save
        :param use_zipfile_serialization_if_available: for torch >= 1.6.0 only
            exports the Module's state dict using the new zipfile serialization
        :param include_modifiers: if True, and a ScheduledOptimizer is provided
            as the optimizer, the associated ScheduledModifierManager and its
            Modifiers will be exported under the 'manager' key. Default is False
        :param export_entire_model: Exports entire file instead of state_dict
        :param arch_key: if provided, the `arch_key` will be saved in the
            checkpoint
        """
        pytorch_path = os.path.join(self._output_dir, "training")
        pth_path = os.path.join(pytorch_path, name)
        create_parent_dirs(pth_path)

        if export_entire_model:
            torch.save(self._module, pth_path)
        else:
            save_model(
                pth_path,
                self._module,
                optimizer,
                recipe,
                epoch,
                use_zipfile_serialization_if_available=(
                    use_zipfile_serialization_if_available
                ),
                include_modifiers=include_modifiers,
                arch_key=arch_key,
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
        inputs_dir = os.path.join(self._output_dir, "sample_inputs")
        outputs_dir = os.path.join(self._output_dir, "sample_outputs")
        labels_dir = os.path.join(self._output_dir, "sample_labels")
        originals_dir = os.path.join(self._output_dir, "sample_originals")

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


def export_onnx(
    module: Module,
    sample_batch: Any,
    file_path: str,
    opset: int = TORCH_DEFAULT_ONNX_OPSET,
    disable_bn_fusing: bool = True,
    convert_qat: bool = False,
    dynamic_axes: Union[str, Dict[str, List[int]]] = None,
    skip_input_quantize: bool = False,
    **export_kwargs,
):
    """
    Export an onnx file for the current module and for a sample batch.
    Sample batch used to feed through the model to freeze the graph for a
    particular execution.

    :param module: torch Module object to export
    :param sample_batch: the batch to export an onnx for, handles creating the
        static graph for onnx as well as setting dimensions
    :param file_path: path to the onnx file to save
    :param opset: onnx opset to use for exported model.
        Default is based on torch version.
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
    :param dynamic_axes: dictionary of input or output names to list of dimensions
        of those tensors that should be exported as dynamic. May input 'batch'
        to set the first dimension of all inputs and outputs to dynamic. Default
        is an empty dict
    :param skip_input_quantize: if True, the export flow will attempt to delete
        the first Quantize Linear Nodes(s) immediately after model input and set
        the model input type to UINT8. Default is False
    :param export_kwargs: kwargs to be passed as is to the torch.onnx.export api
        call. Useful to pass in dyanmic_axes, input_names, output_names, etc.
        See more on the torch.onnx.export api spec in the PyTorch docs:
        https://pytorch.org/docs/stable/onnx.html
    """
    if _PARSED_TORCH_VERSION >= version.parse("1.10.0") and opset < 13 and convert_qat:
        warnings.warn(
            "Exporting onnx with QAT and opset < 13 may result in errors. "
            "Please use opset>=13 with QAT. "
            "See https://github.com/pytorch/pytorch/issues/77455 for more info. "
        )

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
    create_parent_dirs(file_path)

    module = deepcopy(module).cpu()

    with torch.no_grad():
        out = tensors_module_forward(sample_batch, module, check_feat_lab_inp=False)

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
                "input_{}".format(index) for index, _ in enumerate(iter(sample_batch))
            ]
            if isinstance(sample_batch, List):
                sample_batch = tuple(sample_batch)  # torch.onnx.export requires tuple

    if "output_names" not in export_kwargs:
        export_kwargs["output_names"] = _get_output_names(out)

    if dynamic_axes is not None:
        warnings.warn(
            "`dynamic_axes` is deprecated and does not affect anything. "
            "The 0th axis is always treated as dynamic.",
            category=DeprecationWarning,
        )

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
    batch_norms_wrapped = False
    if (
        _PARSED_TORCH_VERSION >= version.parse("1.7")
        and not is_quant_module
        and disable_bn_fusing
    ):
        # prevent batch norm fusing by adding a trivial operation before every
        # batch norm layer
        batch_norms_wrapped = _wrap_batch_norms(module)

    kwargs = dict(
        model=module,
        args=sample_batch,
        f=file_path,
        verbose=False,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        **export_kwargs,
    )

    if _PARSED_TORCH_VERSION < version.parse("1.10.0"):
        kwargs["strip_doc_string"] = True
    else:
        kwargs["training"] = torch.onnx.TrainingMode.PRESERVE
        kwargs["do_constant_folding"] = not module.training
        kwargs["keep_initializers_as_inputs"] = False

    torch.onnx.export(**kwargs)

    # re-enable disabled quantization observers
    for submodule in disabled_observers:
        submodule.observer_enabled[0] = 1

    # onnx file fixes
    onnx_model = onnx.load(file_path)
    _fold_identity_initializers(onnx_model)
    _flatten_qparams(onnx_model)
    if batch_norms_wrapped:
        # fix changed batch norm names
        _unwrap_batchnorms(onnx_model)

        # clean up graph from any injected / wrapped operations
        _delete_trivial_onnx_adds(onnx_model)
    onnx.save(onnx_model, file_path)

    if convert_qat and is_quant_module:
        # overwrite exported model with fully quantized version
        # import here to avoid cyclic dependency
        from sparseml.pytorch.sparsification.quantization import (
            quantize_torch_qat_export,
        )

        use_qlinearconv = hasattr(module, "export_with_qlinearconv") and (
            module.export_with_qlinearconv
        )

        quantize_torch_qat_export(
            model=file_path,
            output_file_path=file_path,
            use_qlinearconv=use_qlinearconv,
        )

    if skip_input_quantize:
        try:
            # import here to avoid cyclic dependency
            from sparseml.pytorch.sparsification.quantization import (
                skip_onnx_input_quantize,
            )

            skip_onnx_input_quantize(file_path, file_path)
        except Exception as e:
            _LOGGER.warning(
                f"Unable to skip input QuantizeLinear op with exception {e}"
            )


def _copy_file(src: str, target: str):
    if not os.path.exists(src):
        raise ValueError(
            f"Attempting to copy file from {src}, but the file does not exist."
        )
    shutil.copyfile(src, target)


def _create_config_file(save_dir: str) -> str:
    config_file_path = os.path.join(save_dir, CONFIG_JSON_NAME)
    with open(config_file_path, "w"):
        # create empty json file
        pass

    _LOGGER.info(f"Created {CONFIG_JSON_NAME} file at {save_dir}")
    return config_file_path


def _save_label_to_class_mapping(
    labels_to_class_mapping: Union[str, Dict[int, str]],
    config_file_path: str,
    key_name: str = "labels_to_class_mapping",
):
    """
    Appends `labels_to_class_mapping` information to the config.json file:
        - new key: `labels_to_class_mapping`
        - new value: a dictionary that maps the integer
          labels to string class names
    If config.json already contains `labels_to_class_mapping`,
    this information will be overwritten

    :param labels_to_class_mapping: information about the mapping from
        integer labels to string class names. Can be either a string
        (path to the .json serialized dictionary) or a dictionary.
    :param config_file_path: path to the directory of the `config.json` file.
    :param key_name: the key under which the information about
        the mapping will be stored inside the config.json file
    """
    is_config_empty = os.stat(config_file_path).st_size == 0

    if not is_config_empty:
        with open(config_file_path, "r") as outfile:
            config = json.load(outfile.read())
    else:
        config = {}

    # check whether the label names are not already present in the config.
    if key_name in config.keys():
        _LOGGER.warning(
            f"File: {CONFIG_JSON_NAME} already contains key {key_name}. "
            f"{key_name} data will be overwritten"
        )

    if isinstance(labels_to_class_mapping, str):
        with open(labels_to_class_mapping) as outfile:
            labels_to_class_mapping = json.load(outfile)

    config[key_name] = labels_to_class_mapping

    with open(config_file_path, "w") as outfile:
        json.dump(config, outfile)

    _LOGGER.info(
        f"Appended {key_name} data to {CONFIG_JSON_NAME} at {config_file_path}"
    )


def _flatten_qparams(model: onnx.ModelProto):
    # transforms any QuantizeLinear/DequantizeLinear that have
    # zero_point/scale with shapes `(1,)` into shape `()`
    graph = ONNXGraph(model)

    inits_to_flatten = set()

    for node in model.graph.node:
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            # scale is required if the input is an initializer
            scale_init = graph.get_init_by_name(node.input[1])
            if scale_init is not None and list(scale_init.dims) == [1]:
                inits_to_flatten.add(node.input[1])

                # zero_point is optional AND shape must match
                # scale. so if scale is (1,), then so will zero point
                if len(node.input) == 3:
                    inits_to_flatten.add(node.input[2])

    for i, init in enumerate(model.graph.initializer):
        if init.name not in inits_to_flatten:
            continue
        a = numpy_helper.to_array(init)
        assert a.shape == (1,)
        b = numpy.array(a[0])
        assert b.shape == ()
        assert b.dtype == a.dtype
        model.graph.initializer[i].CopyFrom(numpy_helper.from_array(b, name=init.name))


def _fold_identity_initializers(model: onnx.ModelProto):
    # folds any Identity nodes that have a single input (which is an initializer)
    # and a single output
    matches = []

    graph = ONNXGraph(model)

    def is_match(node: onnx.NodeProto) -> bool:
        return (
            node.op_type == "Identity"
            and len(node.input) == 1
            and len(node.output) == 1
            and node.input[0] in graph._name_to_initializer
        )

    for node in model.graph.node:
        if not is_match(node):
            continue
        matches.append(node)

        # find any node in the graph that uses the output of `node`
        # as an input. replace the input with `node`'s input
        for other in graph.get_node_children(node):
            for i, other_input_i in enumerate(other.input):
                # NOTE: this just replaces the str ids
                if other_input_i == node.output[0]:
                    other.input[i] = node.input[0]

    for node in matches:
        model.graph.node.remove(node)


def _get_output_names(out: Any):
    """
    Get name of output tensors

    :param out: outputs of the model
    :return: list of names
    """
    output_names = None
    if isinstance(out, Tensor):
        output_names = ["output"]
    elif hasattr(out, "keys") and callable(out.keys):
        output_names = list(out.keys())
    elif isinstance(out, Iterable):
        output_names = ["output_{}".format(index) for index, _ in enumerate(iter(out))]
    return output_names


class _AddNoOpWrapper(Module):
    # trivial wrapper to break-up Conv-BN blocks

    def __init__(self, module: Module):
        super().__init__()
        self.bn_wrapper_replace_me = module

    def forward(self, inp):
        inp = inp + 0  # no-op
        return self.bn_wrapper_replace_me(inp)


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


def _unwrap_batchnorms(model: onnx.ModelProto):
    for init in model.graph.initializer:
        init.name = init.name.replace(".bn_wrapper_replace_me", "")
    for node in model.graph.node:
        for idx in range(len(node.input)):
            node.input[idx] = node.input[idx].replace(".bn_wrapper_replace_me", "")
        for idx in range(len(node.output)):
            node.output[idx] = node.output[idx].replace(".bn_wrapper_replace_me", "")

    onnx.checker.check_model(model)
