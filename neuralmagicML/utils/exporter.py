from typing import Union, Tuple, List
import os
import numpy
import torch
from torch import Tensor
from torch.nn import Module


__all__ = ['ModelExporter']


class ModelExporter(object):
    def __init__(self, model: Module, inputs_shape: Union[List[Tuple[int, ...]], Tuple[int, ...]], output_dir: str):
        self._model = model.to('cpu').eval()
        self._inputs_shape = inputs_shape if isinstance(inputs_shape, List) else [inputs_shape]
        self._output_dir = os.path.abspath(os.path.expanduser(output_dir))

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

    def export_onnx(self):
        onnx_path = os.path.join(self._output_dir, 'model.onnx')
        inp = self._create_sample_input(batch_size=1)

        with torch.no_grad():
            out = self._model(*inp)

        input_names = ['input'] if len(inp) == 1 else ['input_{}'.format(ind) for ind in range(len(inp))]
        output_names = ['output'] if isinstance(out, Tensor) or len(out) == 1 \
            else ['output_{}'.format(ind) for ind in range(len(out))]

        torch.onnx.export(self._model, inp, onnx_path, input_names=input_names, output_names=output_names,
                          strip_doc_string=True, verbose=False)
        print('Exported onnx to {}'.format(onnx_path))

    def export_batch(self, batch_size: int, export_intermediates: bool = False, export_layers: bool = False):
        batch_dir = os.path.join(self._output_dir, 'b{}'.format(batch_size))
        tensors_dir = os.path.join(batch_dir, 'tensors')
        layers_dir = os.path.join(batch_dir, 'layers')

        if not os.path.exists(tensors_dir):
            os.makedirs(tensors_dir)

        if not os.path.exists(layers_dir):
            os.makedirs(layers_dir)

        handles = []
        layer_type_counts = {}

        def forward_hook(_layer: Module, _inp: Tuple[Tensor, ...], _out: Union[Tensor, Tuple[Tensor, ...]]):
            type_ = type(_layer).__name__

            if type_ not in layer_type_counts:
                layer_type_counts[type_] = 0

            export_name = '{}-{}.{}'.format(type_, layer_type_counts[type_], mod.__layer_name.replace('.', '-'))
            layer_type_counts[type_] += 1

            if export_layers:
                ModelExporter.export_layer(_layer, export_name, layers_dir)

            if export_intermediates:
                ModelExporter.export_tensor(_inp, '{}.input'.format(export_name), tensors_dir)
                ModelExporter.export_tensor(_out, '{}.output'.format(export_name), tensors_dir)

        for name, mod in self._model.named_modules():
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
            inp = self._create_sample_input(batch_size)
            out = self._model(*inp)

        ModelExporter.export_tensor(inp, '_model.input', tensors_dir)
        ModelExporter.export_tensor(out, '_model.output', tensors_dir)

    def _create_sample_input(self, batch_size: int) -> Tuple[Tensor]:
        return tuple(torch.randn(batch_size, *inp) for inp in self._inputs_shape)

    @staticmethod
    def export_layer(layer: Module, name: str, export_dir: str):
        for param_name, param in layer.named_parameters():
            export_path = os.path.join(export_dir, '{}.{}.npy'.format(name, param_name))
            export_tens = param.data.cpu().detach().numpy()
            numpy.save(export_path, export_tens)
            print('exported param {} for layer {} to {}'.format(param_name, name, export_path))

    @staticmethod
    def export_tensor(tensors: Union[Tensor, Tuple[Tensor, ...]], name: str, export_dir: str):
        if isinstance(tensors, Tensor):
            tensors = (tensors,)

        for index, tens in enumerate(tensors):
            export_path = os.path.join(export_dir, '{}-{}.npy'.format(name, index))
            export_tens = tens.cpu().detach().numpy()
            numpy.save(export_path, export_tens)
            print('exported tensor #{} for {} to {}'.format(index, name, export_path))
