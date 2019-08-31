from typing import Union, List, Tuple
import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy
import pandas
import torch
from torch.nn import ReLU

from neuralmagicML.datasets import create_dataset, EarlyStopDataset
from neuralmagicML.models import create_model
from neuralmagicML.sparsity import convert_to_bool, ModuleBoostAnalysis, ASAnalyzerLayer
from neuralmagicML.utils import CrossEntropyLossWrapper, TopKAccuracy, ModuleTestResults


def as_boost_analysis(device_desc: str, model_type: str, pretrained: Union[bool, str], model_path: Union[None, str],
                      dataset_type: str, dataset_root: str, sample_size: int, train_size: Union[float, int],
                      train_batch_size: int, test_batch_size: int, num_workers: int, pin_memory: bool,
                      lr: float, momentum: float, dampening: float, weight_decay: float, nesterov: bool,
                      save_dir: str, debug_early_stop: int):
    ####################################################################################################################
    #
    # Dataset setup section
    #
    ####################################################################################################################
    if train_size > 0:
        train_dataset, num_classes = create_dataset(dataset_type, dataset_root, train=True, rand_trans=True)
        if debug_early_stop > 0:
            train_dataset = EarlyStopDataset(train_dataset, debug_early_stop)
        print('created train dataset: \n{}\n'.format(train_dataset))
    else:
        train_dataset = None

    val_dataset, num_classes = create_dataset(dataset_type, dataset_root, train=False, rand_trans=False)
    if debug_early_stop > 0:
        val_dataset = EarlyStopDataset(val_dataset, debug_early_stop)
    print('created val dataset: \n{}\n'.format(val_dataset))

    ####################################################################################################################
    #
    # Model and loss function setup
    #
    ####################################################################################################################

    model = create_model(model_type, model_path, pretrained=pretrained, num_classes=num_classes)
    print('Created model of type {} with num_classes:{}, pretrained:{} / model_path:{}'
          .format(model_type, num_classes, pretrained, model_path))

    # only support for now is cross entropy with top1 and top5 accuracy
    loss = CrossEntropyLossWrapper(
        extras={'top1acc': TopKAccuracy(1), 'top5acc': TopKAccuracy(5)}
    )
    print('Created loss {}'.format(loss))

    ####################################################################################################################
    #
    # Analyzing section
    #
    ####################################################################################################################

    save_dir = os.path.abspath(os.path.expanduser(save_dir))
    model_tag = '{}-{}'.format(model_type.replace('/', '.'), dataset_type)
    model_id = model_tag
    model_inc = 0

    while os.path.exists(os.path.join(save_dir, model_id)):
        model_inc += 1
        model_id = '{}__{:02d}'.format(model_tag, model_inc)

    model_dir = os.path.join(save_dir, model_id)
    os.makedirs(model_dir)
    print('Model dir set to {}'.format(model_dir))

    analysis = ModuleBoostAnalysis(model, device_desc, loss, val_dataset, test_batch_size,
                                   sample_size, train_dataset, train_batch_size, train_size,
                                   dataloader_num_workers=num_workers, dataloader_pin_memory=pin_memory,
                                   optim_lr=lr, optim_momentum=momentum, optim_dampening=dampening,
                                   optim_weight_decay=weight_decay, optim_nesterov=nesterov)
    relu_layers = []

    for name, mod in model.named_modules():
        if isinstance(mod, ReLU):
            relu_layers.append(name)

    for layer in relu_layers:
        layer_results = analysis.analyze_layer(layer)
        _save_layer_results(layer, layer_results, model_dir)

    print('Completed')


def _save_layer_results(layer: str, results: List[Tuple[float, ModuleTestResults, ASAnalyzerLayer]], model_dir):
    json_data = {
        'inputs': torch.cat(results[0][2].inputs_sample).tolist(),
        'thresholds': {}
    }

    for (thresh, test_results, analyzer_layer) in results:
        test_results = test_results  # type: ModuleTestResults
        json_data['thresholds'][thresh] = {
            'sparsities': torch.cat(analyzer_layer.outputs_sparsity).tolist(),
            'losses': {loss: torch.cat(values).tolist() for loss, values in test_results.results.items()}
        }

    json_path = os.path.join(model_dir, '{}.json'.format(layer))

    with open(json_path, 'w') as layers_file:
        json.dump(json_data, layers_file)

    print('saved layer {} results json to {}'.format(layer, json_path))

    inputs = torch.cat(results[0][2].inputs_sample).view(-1).tolist()
    thresholds = [res[0] for res in results]
    sparsities = [res[2].outputs_sparsity_mean.item() * 100.0 for res in results]
    losses = {}

    for res in results:
        for loss in res[1].results.keys():
            if loss not in losses:
                losses[loss] = []

            losses[loss].append(res[1].result_mean(loss).item())

    size = (10, 7)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=size, gridspec_kw={'width_ratios': [1, 1]})

    text_axis = axes[0, 0]
    text_axis.axis('off')
    text_axis.text(0.5, 1.0, layer, horizontalalignment='center', verticalalignment='top',
                   fontsize=16, fontweight='bold')

    inputs_axis = axes[0, 1]
    inputs_frame = pandas.DataFrame(inputs, columns=['inputs'])
    inputs_frame.hist(column='inputs', bins=256, ax=inputs_axis)
    inputs_axis.set_title('Baseline Inputs Distribution')
    inputs_axis.set_ylabel('Count')
    inputs_axis.set_xlabel('Input Value')

    loss_axis = axes[1, 0]
    loss_frame = pandas.DataFrame(data={
        'loss': losses[list(losses.keys())[0]]
    }, index=thresholds)
    loss_frame.plot.line(ax=loss_axis)
    loss_axis.set_title('Loss vs Threshold')
    loss_axis.set_ylabel('Value')
    loss_axis.set_xlabel('Threshold')
    plt.setp(loss_axis.get_xticklabels(), rotation=30, horizontalalignment='right')

    sparsity_axis = axes[1, 1]
    sparsity_frame = pandas.DataFrame(data={
        **losses,
        'sparsity': sparsities
    }, index=thresholds)
    sparsity_frame.plot.line(ax=sparsity_axis)
    sparsity_axis.set_title('Sparsity / Losses vs Threshold')
    sparsity_axis.set_ylabel('Value')
    sparsity_axis.set_xlabel('Threshold')
    plt.setp(sparsity_axis.get_xticklabels(), rotation=30, horizontalalignment='right')

    fig_path = os.path.join(model_dir, '{}.png'.format(layer))
    plt.savefig(fig_path)
    plt.close(fig)
    print('saved layer {} figure to {}'.format(layer, fig_path))


def main():
    parser = argparse.ArgumentParser(description='Analyze the boosted activation sparsity for a model '
                                                 'compared with its the loss for sensitivity')

    # device and model arguments
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

    # dataset settings
    parser.add_argument('--dataset-type', type=str, required=True,
                        help='The dataset type to load for training, ex: imagenet, imagenette, cifar10, etc')
    parser.add_argument('--dataset-root', type=str, required=True,
                        help='The root path to where the dataset is stored')
    parser.add_argument('--sample-size', type=int, required=True,
                        help='The total number of samples to run through for calculating losses and sparsity values')
    parser.add_argument('--train-size', type=float, default=-1.0,
                        help='If > 1 then the number of data items to run through training after each threshold step '
                             'If > 0 and <= 1 then the percent of the train dataset to run through after each step '
                             'If <= 0 then will not run training')
    parser.add_argument('--train-batch-size', type=int, required=True,
                        help='The batch size to use while training')
    parser.add_argument('--test-batch-size', type=int, required=True,
                        help='The batch size to use while testing')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='The number of workers to use for data loading')
    parser.add_argument('--pin-memory', type=bool, default=True,
                        help='Use pinned memory for data loading')

    # optimizer settings
    parser.add_argument('--load-optim', type=bool, default=False,
                        help='Load the previous optimizer state from the model file (restore from checkpoint)')
    parser.add_argument('--lr', type=float, required=True,
                        help='The learning rate to use while training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='The momentum to use for the SGD optimizer')
    parser.add_argument('--dampening', type=float, default=0.0,
                        help='The dampening to use for the SGD optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='The l2 regularization to apply to the weights')
    parser.add_argument('--nesterov', type=bool, default=True,
                        help='Use nesterov momentum for the SGD optimizer')

    # save options
    parser.add_argument('--save-dir', type=str, default='as-boost-analysis',
                        help='The path to the directory for saving results')

    # debug options
    parser.add_argument('--debug-early-stop', type=int, default=-1,
                        help='Early stop going through the datasets to debug issues, '
                             'will create the datasets with this number of items')

    args = parser.parse_args()
    as_boost_analysis(
        args.device, args.model_type, args.pretrained, args.model_path,
        args.dataset_type, args.dataset_root, args.sample_size, args.train_size,
        args.train_batch_size, args.test_batch_size, args.num_workers, convert_to_bool(args.pin_memory),
        args.lr, args.momentum, args.dampening, args.weight_decay, convert_to_bool(args.nesterov),
        args.save_dir,
        args.debug_early_stop
    )


if __name__ == '__main__':
    main()
