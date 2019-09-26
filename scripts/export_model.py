from typing import Union, List, Dict
import argparse
import os
import json

from neuralmagicML.models import create_model
from neuralmagicML.sparsity import convert_to_bool
from neuralmagicML.utils import ModelExporter, fix_onnx_threshold_export


def export_model(model_type: str, pretrained: Union[bool, str], model_path: Union[None, str], num_classes: int,
                 export_dir: Union[None, str], export_batch_sizes: str, export_input_sizes: str,
                 export_intermediates: bool, export_layers: bool,
                 model_plugin_paths: Union[None, List[str]], model_plugin_args: Union[None, Dict]):
    # fix for exporting FATReLU's
    fix_onnx_threshold_export()

    if export_dir is None:
        export_dir = os.path.join('.', 'onnx', model_type)

    export_dir = os.path.abspath(os.path.expanduser(export_dir))

    model = create_model(model_type, model_path, plugin_paths=model_plugin_paths, plugin_args=model_plugin_args,
                         pretrained=pretrained, num_classes=num_classes)
    print('Created model of type {} with num_classes:{}, pretrained:{} / model_path:{}, and plugins:{}'
          .format(model_type, num_classes, pretrained, model_path, model_plugin_paths))

    export_batch_sizes = [int(batch_size) for batch_size in export_batch_sizes.split(',')]
    export_input_sizes = [tuple([int(size) for size in input_size.split(',')])
                          for input_size in export_input_sizes.split(';')]

    print('Exporting model to {} for input_sizes:{}, batch_sizes:{}, intermediates:{}, and layers:{}'
          .format(export_dir, export_batch_sizes, export_input_sizes, export_intermediates, export_layers))

    exporter = ModelExporter(model, export_input_sizes, export_dir)
    exporter.export_onnx()

    for batch_size in export_batch_sizes:
        exporter.export_batch(batch_size, export_intermediates, export_layers)

    print('Completed')


def main():
    parser = argparse.ArgumentParser(description='Export a model for a given dataset with ')

    # schedule device and model arguments
    parser.add_argument('--model-type', type=str, required=True,
                        help='The type of model to create, ex: resnet50, vgg16, mobilenet '
                             'put as help to see the full list (will raise an exception with the list)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='The type of pretrained weights to use, default is not to load any, '
                             'ex: imagenet/dense, imagenet/sparse, imagenette/dense')
    parser.add_argument('--model-path', type=str, default=None,
                        help='The path to the model file to load a previous state dict from')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='The number of classes to create the model for')

    # export settings
    parser.add_argument('--export-dir', type=str, default=None,
                        help='The directory to export the model under, defaults to onnx/MODEL_TYPE')
    parser.add_argument('--export-batch-sizes', type=str, default='1',
                        help='A comma separated list of the batch sizes to export, ex: 1,16,64')
    parser.add_argument('--export-input-sizes', type=str, default='3,224,224',
                        help='A comma and semi colon separated list for the inputs to use to the model '
                             'commas separate the shapes within a single input, semi colon separates multi input '
                             'ex: 3,224,224   3,224,224;3,128,128')
    parser.add_argument('--export-intermediates', type=bool, default=False,
                        help='Export the intermediate tensors within the model execution')
    parser.add_argument('--export-layers', type=bool, default=False,
                        help='Export the layer params for the model to numpy files')

    # plugin options
    parser.add_argument('--model-plugin-paths', default=None, nargs='+',
                        help='plugins to load for handling a model and possibly teacher after creation')
    parser.add_argument('--model-plugin-args', type=json.loads, default={},
                        help='json string containing the args to pass to the model plugins when executing')

    args = parser.parse_args()
    export_model(
        args.model_type, args.pretrained, args.model_path, args.num_classes,
        args.export_dir, args.export_batch_sizes, args.export_input_sizes,
        convert_to_bool(args.export_intermediates), convert_to_bool(args.export_layers),
        args.model_plugin_paths, args.model_plugin_args
    )


if __name__ == '__main__':
    main()
