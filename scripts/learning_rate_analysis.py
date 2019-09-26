from typing import Union, Dict, List
import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from torch.nn import Module

from neuralmagicML.datasets import create_dataset
from neuralmagicML.models import create_model
from neuralmagicML.sparsity import convert_to_bool
from neuralmagicML.utils import CrossEntropyLossWrapper, lr_analysis, lr_analysis_figure


def learning_rate_analysis(
        batches_per_sample: int, device_desc: str,
        model_type: str, pretrained: Union[bool, str], model_path: Union[None, str],
        model_plugin_paths: Union[None, List[str]], model_plugin_args: Union[None, Dict],
        dataset_type: str, dataset_root: str, train_batch_size: int, num_workers: int, pin_memory: bool,
        init_lr: float, final_lr: float, momentum: float, dampening: float, weight_decay: float, nesterov: bool,
        save_dir: str):

    train_dataset, num_classes = create_dataset(dataset_type, dataset_root, train=True, rand_trans=True)
    train_data_loader = DataLoader(train_dataset, train_batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=pin_memory)

    model = create_model(model_type, model_path, plugin_paths=model_plugin_paths, plugin_args=model_plugin_args,
                         pretrained=pretrained, num_classes=num_classes)  # type: Module
    print('Created model of type {} with num_classes:{}, pretrained:{} / model_path:{}, and plugins:{}'
          .format(model_type, num_classes, pretrained, model_path, model_plugin_paths))

    loss = CrossEntropyLossWrapper()
    print('Created loss {}'.format(loss))

    save_dir = os.path.abspath(os.path.expanduser(save_dir))
    model_tag = '{}-{}'.format(model_type.replace('/', '.'), dataset_type)
    model_id = model_tag
    model_inc = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    while os.path.exists(os.path.join(save_dir, model_id)):
        model_inc += 1
        model_id = '{}__{:02d}'.format(model_tag, model_inc)

    print('Model id is set to {}'.format(model_id))
    save_path = os.path.join(save_dir, '{}.png'.format(model_id))

    print('Running LR analysis...')
    analysis = lr_analysis(model, device_desc, train_data_loader, CrossEntropyLossWrapper(), batches_per_sample,
                           init_lr=init_lr, final_lr=final_lr,
                           sgd_momentum=momentum, sgd_dampening=dampening,
                           sgd_weight_decay=weight_decay, sgd_nesterov=nesterov)
    fig, axes = lr_analysis_figure(analysis)
    fig.savefig(save_path)
    print('Saved lr analysis to {}'.format(save_path))


def main():
    parser = argparse.ArgumentParser(description='Train a model for a given dataset with a given schedule')

    # schedule device and model arguments
    parser.add_argument('--batches-per-sample', type=int, default=10,
                        help='The number of batches to run for each learning rate sample')
    parser.add_argument('--device', type=str, default='cpu' if not torch.cuda.is_available() else 'cuda',
                        help='The device to run on (can also include ids for data parallel), ex: '
                             'cpu, cuda, cuda:0,1')
    parser.add_argument('--model-type', type=str, required=True,
                        help='The type of model to create, ex: resnet50, vgg16, mobilenet'
                             'put as help to see the full list (will raise an exception with the list)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='The type of pretrained weights to use, default is not to load any, '
                             'ex: imagenet/dense, imagenet/sparse, imagenette/dense')
    parser.add_argument('--model-path', type=str, default=None,
                        help='The path to the model file to load a previous state dict from')

    # plugin options
    parser.add_argument('--model-plugin-paths', default=None, nargs='+',
                        help='plugins to load for handling a model and possibly teacher after creation')
    parser.add_argument('--model-plugin-args', type=json.loads, default={},
                        help='json string containing the args to pass to the model plugins when executing')

    # dataset settings
    parser.add_argument('--dataset-type', type=str, required=True,
                        help='The dataset type to load for training, ex: imagenet, imagenette, cifar10, etc')
    parser.add_argument('--dataset-root', type=str, required=True,
                        help='The root path to where the dataset is stored')
    parser.add_argument('--train-batch-size', type=int, required=True,
                        help='The batch size to use while training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='The number of workers to use for data loading')
    parser.add_argument('--pin-memory', type=bool, default=True,
                        help='Use pinned memory for data loading')

    # optimizer settings
    parser.add_argument('--init-lr', type=float, default=10e-6,
                        help='The initial learning rate to use while training')
    parser.add_argument('--final-lr', type=float, default=0.5,
                        help='The initial learning rate to use while training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='The momentum to use for the SGD optimizer')
    parser.add_argument('--dampening', type=float, default=0.0,
                        help='The dampening to use for the SGD optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='The l2 regularization to apply to the weights')
    parser.add_argument('--nesterov', type=bool, default=True,
                        help='Use nesterov momentum for the SGD optimizer')

    # save options
    parser.add_argument('--save-dir', type=str, default='lr-analysis',
                        help='The path to the directory for saving results')

    args = parser.parse_args()
    learning_rate_analysis(
        args.batches_per_sample, args.device,
        args.model_type, args.pretrained, args.model_path,
        args.model_plugin_paths, args.model_plugin_args,
        args.dataset_type, args.dataset_root,
        args.train_batch_size, args.num_workers, convert_to_bool(args.pin_memory),
        args.init_lr, args.final_lr,
        args.momentum, args.dampening, args.weight_decay, convert_to_bool(args.nesterov),
        args.save_dir
    )


if __name__ == '__main__':
    main()
