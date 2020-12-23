"""
Export PyTorch models to the local device
"""

from typing import List, Any, Iterable
import os
from copy import deepcopy

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from neuralmagicML.utils import (
    create_parent_dirs,
    clean_path,
)
from neuralmagicML.pytorch.utils.helpers import (
    tensors_to_device,
    tensors_module_forward,
    tensors_export,
)
from neuralmagicML.pytorch.utils.model import (
    save_model,
    is_parallel_model,
)


__all__ = ["ModuleExporter"]


DEFAULT_ONNX_OPSET = 9 if torch.__version__ < "1.3" else 11


class ModuleExporter(object):
    """
    An exporter for exporting PyTorch modules into ONNX format
    as well as numpy arrays for the input and output tensors.

    :param module: the module to export
    :param output_dir: the directory to export the module and extras to
    """

    def __init__(
        self, module: Module, output_dir: str,
    ):
        if is_parallel_model(module):
            module = module.module

        self._module = deepcopy(module).to("cpu").eval()
        self._output_dir = clean_path(output_dir)

    def export_onnx(
        self,
        sample_batch: Any,
        name: str = "model.onnx",
        opset: int = DEFAULT_ONNX_OPSET,
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
        """
        sample_batch = tensors_to_device(sample_batch, "cpu")
        onnx_path = os.path.join(self._output_dir, name)
        create_parent_dirs(onnx_path)

        with torch.no_grad():
            out = tensors_module_forward(sample_batch, self._module)

        input_names = None
        if isinstance(sample_batch, Tensor):
            input_names = ["input"]
        elif isinstance(sample_batch, Iterable):
            input_names = [
                "input_{}".format(index) for index, _ in enumerate(iter(sample_batch))
            ]

        output_names = None
        if isinstance(out, Tensor):
            output_names = ["output"]
        elif isinstance(out, Iterable):
            output_names = [
                "output_{}".format(index) for index, _ in enumerate(iter(out))
            ]

        # disable active quantization observers because they cannot be exported
        disabled_observers = []
        for submodule in self._module.modules():
            if (
                hasattr(submodule, "observer_enabled")
                and submodule.observer_enabled[0] == 1
            ):
                submodule.observer_enabled[0] = 0
                disabled_observers.append(submodule)

        torch.onnx.export(
            self._module,
            sample_batch,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            strip_doc_string=True,
            verbose=False,
            opset_version=opset,
        )

        # re-enable disabled quantization observers
        for submodule in disabled_observers:
            submodule.observer_enabled[0] = 1

    def export_pytorch(
        self,
        optimizer: Optimizer = None,
        epoch: int = None,
        name: str = "model.pth",
        use_zipfile_serialization_if_available: bool = True,
    ):
        """
        Export the pytorch state dicts into pth file within a
        pytorch framework directory.

        :param optimizer: optional optimizer to export along with the module
        :param epoch: optional epoch to export along with the module
        :param name: name of the pytorch file to save
        :param use_zipfile_serialization_if_available: for torch >= 1.6.0 only
            exports the Module's state dict using the new zipfile serialization
        """
        pytorch_path = os.path.join(self._output_dir, "pytorch")
        pth_path = os.path.join(pytorch_path, name)
        create_parent_dirs(pth_path)
        save_model(
            pth_path,
            self._module,
            optimizer,
            epoch,
            use_zipfile_serialization_if_available=use_zipfile_serialization_if_available,
        )

    def export_samples(
        self,
        sample_batches: List[Any],
        sample_labels: List[Any] = None,
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
        inputs_dir = os.path.join(self._output_dir, "_sample-inputs")
        outputs_dir = os.path.join(self._output_dir, "_sample-outputs")
        labels_dir = os.path.join(self._output_dir, "_sample-labels")

        with torch.no_grad():
            for batch, lab in zip(
                sample_batches,
                sample_labels if sample_labels else [None for _ in sample_batches],
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
                    exported_label = tensors_export(
                        lab, labels_dir, "lab", counter=exp_counter, break_batch=True
                    )

                assert len(exported_input) == len(exported_output)
                exp_counter += len(exported_input)
