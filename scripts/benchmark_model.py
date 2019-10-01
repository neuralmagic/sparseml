from typing import Union, List, Dict, Tuple
import argparse
import os
import json
import csv
import torch

from neuralmagicML.models import create_model
from neuralmagicML.utils import ModelBenchmarker, BenchmarkResults


def benchmark_model(device: str,
                    model_type: str, pretrained: Union[bool, str], model_path: Union[None, str], num_classes: int,
                    model_plugin_paths: Union[None, List[str]], model_plugin_args: Union[None, Dict],
                    output_dir: str, batch_sizes: List[int], input_sizes: List[Tuple[int]],
                    warmup_size: int, test_size: int):
    model = create_model(model_type, model_path, plugin_paths=model_plugin_paths, plugin_args=model_plugin_args,
                         pretrained=pretrained, num_classes=num_classes)
    print('Created model of type {} with num_classes:{}, pretrained:{} / model_path:{}, and plugins:{}'
          .format(model_type, num_classes, pretrained, model_path, model_plugin_paths))

    benchmarker = ModelBenchmarker(model_type, model, device)
    results = []

    print('Running benchmarks for batch_sizes:{} and input_sizes:{}'.format(batch_sizes, input_sizes))

    for batch_size in batch_sizes:
        print('Benchmarking batch_size: {}'.format(batch_size))
        batch = [torch.randn(batch_size, *inp) for inp in input_sizes]
        res = benchmarker.benchmark_batch(batch, full_precision=True, test_size=test_size, warmup_size=warmup_size)
        results.append((batch_size, res))

    _save_results(results, output_dir, model_type, device, batch_sizes)

    print('Completed')


def _save_results(results: List[Tuple[int, BenchmarkResults]], output_dir: str, model_type: str, device: str,
                  batch_sizes: List[int]):
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_id = '{}-{}-{}'.format(model_type, device, ','.join([str(size) for size in batch_sizes]))
    model_tag = model_id
    model_inc = 0

    if os.path.exists(os.path.join(output_dir, '{}.json'.format(model_id))):
        model_inc += 1
        model_tag = '{}_{}'.format(model_id, model_inc)

    save_path = os.path.join(output_dir, model_tag)
    batch_results = {}

    for (batch, res) in results:
        batch_results['b{}'.format(batch)] = {
            'e2e_sec_img': res.e2e_sec_per_item,
            'e2e_ms_img': res.e2e_ms_per_item,
            'e2e_img_sec': res.e2e_items_per_second,
            'model_sec_img': res.model_sec_per_item,
            'model_ms_img': res.model_ms_per_item,
            'model_img_sec': res.model_items_per_second
        }

    save_results = {
        model_type: batch_results
    }

    with open('{}.json'.format(save_path), 'w') as json_file:
        json.dump(save_results, json_file, indent=4)

    print('saved json results to {}.json'.format(save_path))

    with open('{}.csv'.format(save_path), 'w') as csv_file:
        writer = csv.writer(csv_file)
        first_result = batch_results[list(batch_results.keys())[0]]
        writer.writerow(['batch_size', *list(first_result.keys())])

        for batch_size, values in batch_results.items():
            writer.writerow([batch_size, *list(values.values())])

    print('saved csv results to {}.csv'.format(save_path))
    print('')
    print('{} results:'.format(model_type))

    for batch_size, values in batch_results.items():
        print('batch_size: {}'.format(batch_size))

        for key, val in values.items():
            print('\t{}: {}'.format(key, val))


def main():
    parser = argparse.ArgumentParser(description='Benchmark a model for given batch and input sizes')

    # model arguments
    parser.add_argument('--device', type=str, default='cpu' if not torch.cuda.is_available() else 'cuda',
                        help='The device to run on (can also include ids for data parallel), ex: '
                             'cpu, cuda, cuda:0,1')
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

    # plugin options
    parser.add_argument('--model-plugin-paths', default=None, nargs='+',
                        help='plugins to load for handling a model and possibly teacher after creation')
    parser.add_argument('--model-plugin-args', type=json.loads, default={},
                        help='json string containing the args to pass to the model plugins when executing')

    # benchmark options
    parser.add_argument('--output-dir', type=str, default='benchmarks',
                        help='The directory to output the model under, defaults to benchmarks')
    parser.add_argument('--batch-sizes', type=int, default=[1], nargs='+',
                        help='The batch sizes to run, multiple can be supplied for the argument, '
                             'ex: 1  ;  1 8')
    parser.add_argument('--input-sizes', type=str, default=['3,224,224'], nargs='+',
                        help='A comma separated list for the inputs to use to the model, '
                             'multiple inputs can be supplied for the argument '
                             'ex: 3,224,224  ;   3,224,224 3,128,128')
    parser.add_argument('--warmup-size', type=int, default=10,
                        help='The number of warmup runs through the model before tests')
    parser.add_argument('--test-size', type=int, default=100,
                        help='The number of tests to run through the model')

    args = parser.parse_args()
    input_sizes = list(args.input_sizes)

    for index in range(len(input_sizes)):
        input_sizes[index] = tuple(int(inp) for inp in input_sizes[index].split(','))

    benchmark_model(
        args.device,
        args.model_type, args.pretrained, args.model_path, args.num_classes,
        args.model_plugin_paths, args.model_plugin_args,
        args.output_dir, args.batch_sizes, input_sizes,
        args.warmup_size, args.test_size
    )


if __name__ == '__main__':
    main()
