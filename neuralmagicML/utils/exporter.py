"""
Code related to exporting pytorch models on a given device for given batches
"""

from typing import Union, Tuple, List, Any, Iterable
import os
import numpy
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from .helpers import (
    tensors_batch_size,
    tensors_to_device,
    tensors_module_forward,
    tensors_export,
    create_parent_dirs,
    create_dirs,
    clean_path,
)


__all__ = ["ModuleExporter"]


class ModuleExporter(object):
    """
    An exporter for exporting pytorch modules into onnx format
    as well as numpy arrays for the input and output tensors
    additional exports can be numpy arrays for the intermediate activations between layers
    and param values for each layer as numpy arrays
    """

    def __init__(
        self, module: Module, output_dir: str,
    ):
        """
        :param module: the module to export
        :param output_dir: the directory to export the module and extras to
        """
        self._module = module.to("cpu").eval()
        self._output_dir = clean_path(output_dir)

    def export_onnx(self, sample_batch: Any):
        """
        :param sample_batch: the batch to export an onnx for, handles creating the static graph for onnx
                             as well as setting dimensions
        :return: the path to the exported onnx file
        """
        sample_batch = tensors_to_device(sample_batch, "cpu")
        onnx_path = os.path.join(self._output_dir, "model.onnx")
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
                "output_{}".format(index) for index, _ in enumerate(iter(sample_batch))
            ]

        torch.onnx.export(
            self._module,
            sample_batch,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            strip_doc_string=True,
            verbose=False,
        )

        return onnx_path

    def export_batch(
        self,
        sample_batch: Any,
        export_intermediates: bool = False,
        export_layers: bool = False,
    ) -> str:
        """
        :param sample_batch: the batch input to export along with outputs and intermediates and layer params if specified
        :param export_intermediates: True to export the intermediate tensors between layers, False otherwise
        :param export_layers: True to export param values for the layers into numpy arrays, False otherwise
        :return the path to where the batch was exported to
        """
        sample_batch = tensors_to_device(sample_batch, "cpu")
        batch_size = tensors_batch_size(sample_batch)
        batch_dir = os.path.join(self._output_dir, "b{}".format(batch_size))
        tensors_dir = os.path.join(batch_dir, "tensors")
        layers_dir = os.path.join(batch_dir, "layers")
        create_dirs(tensors_dir)
        create_dirs(layers_dir)

        handles = []  # type: List[RemovableHandle]
        layer_type_counts = {}

        def forward_hook(
            _layer: Module,
            _inp: Tuple[Tensor, ...],
            _out: Union[Tensor, Tuple[Tensor, ...]],
        ):
            type_ = type(_layer).__name__

            if type_ not in layer_type_counts:
                layer_type_counts[type_] = 0

            export_name = "{}-{}.{}".format(
                type_, layer_type_counts[type_], _layer.__layer_name.replace(".", "-")
            )
            layer_type_counts[type_] += 1

            if export_layers:
                ModuleExporter.export_layer(_layer, export_name, layers_dir)

            if export_intermediates:
                tensors_export(_inp, tensors_dir, "{}.input".format(export_name))
                tensors_export(_out, tensors_dir, "{}.output".format(export_name))

        for name, mod in self._module.named_modules():
            # make sure we only track root nodes
            # (only has itself in named_modules)
            child_count = 0
            for _, __ in mod.named_modules():
                child_count += 1

            if child_count != 1:
                continue

            mod.__layer_name = name
            handles.append(mod.register_forward_hook(forward_hook))

        with torch.no_grad():
            out = tensors_module_forward(sample_batch, self._module)

        for handle in handles:
            handle.remove()

        handles.clear()
        tensors_export(sample_batch, tensors_dir, "_model.input")
        tensors_export(out, tensors_dir, "_model.output")

        return batch_dir

    @staticmethod
    def export_layer(layer: Module, name: str, export_dir: str):
        for param_name, param in layer.named_parameters():
            export_path = os.path.join(export_dir, "{}.{}.npy".format(name, param_name))
            export_tens = param.data.cpu().detach().numpy()
            numpy.save(export_path, export_tens)
