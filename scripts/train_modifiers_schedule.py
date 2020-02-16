from typing import Union, Dict, List, Iterable, Any
import argparse
import os
import math
import time
import json
from tensorboardX import SummaryWriter
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from torch.nn import Linear, Module
from torch.nn.modules.conv import _ConvNd

from neuralmagicML.datasets import create_dataset, EarlyStopDataset
from neuralmagicML.models import create_model, model_to_device, load_optimizer, save_model
from neuralmagicML.utils import (
    CrossEntropyLossWrapper, TopKAccuracy, KDSettings, ModuleTrainer, ModuleTester, DEFAULT_LOSS_KEY
)
from neuralmagicML.recal import (
    ScheduledModifierManager, ScheduledOptimizer,
    ModuleKSAnalyzer, ModuleASAnalyzer
)
from neuralmagicML.utils import convert_to_bool


def train_modifiers_schedule(
        schedule_path: str, device_desc: str,
        model_type: str, pretrained: Union[bool, str], model_path: Union[None, str],
        teacher_type: Union[str, None], teacher_pretrained: Union[bool, str], teacher_path: Union[None, str],
        kd_temp_student: float, kd_temp_teacher: float, kd_weight: float, kd_contradict_hinton: bool,
        dataset_type: str, dataset_root: str,
        train_batch_size: int, test_batch_size: int, num_workers: int, pin_memory: bool,
        load_optim: bool, init_lr: float, momentum: float, dampening: float, weight_decay: float, nesterov: bool,
        save_dir: str, save_after_epoch: int, save_epochs: Union[None, List[int]], save_epoch_mod: int,
        track_ks: bool, track_as: bool, track_gradients: Union[None, List[str]],
        model_plugin_paths: Union[None, List[str]], model_plugin_args: Union[None, Dict],
        debug_early_stop: int):

    ####################################################################################################################
    #
    # Dataset and data loader setup section
    #
    ####################################################################################################################

    def _create_data_loader(_dataset: Dataset, _train: bool):
        return DataLoader(_dataset, batch_size=train_batch_size if _train else test_batch_size, shuffle=_train,
                          num_workers=num_workers, pin_memory=pin_memory)

    train_dataset, num_classes = create_dataset(dataset_type, dataset_root, train=True, rand_trans=True)
    if debug_early_stop > 0:
        train_dataset = EarlyStopDataset(train_dataset, debug_early_stop)
    print('created train dataset: \n{}\n'.format(train_dataset))

    val_dataset, num_classes = create_dataset(dataset_type, dataset_root, train=False, rand_trans=False)
    if debug_early_stop > 0:
        val_dataset = EarlyStopDataset(val_dataset, debug_early_stop)
    print('created val dataset: \n{}\n'.format(val_dataset))

    train_test_dataset = EarlyStopDataset(
        create_dataset(dataset_type, dataset_root, train=True, rand_trans=False)[0],
        early_stop=len(val_dataset) if len(val_dataset) > 1000 else round(0.2 * len(train_dataset))
    )
    print('created train test dataset: \n{}\n'.format(train_test_dataset))

    ####################################################################################################################
    #
    # Model, knowledge distillation, and loss function setup
    #
    ####################################################################################################################

    model = create_model(model_type, model_path, plugin_paths=model_plugin_paths, plugin_args=model_plugin_args,
                         pretrained=pretrained, num_classes=num_classes)  # type: Module
    print('Created model of type {} with num_classes:{}, pretrained:{} / model_path:{}, and plugins:{}'
          .format(model_type, num_classes, pretrained, model_path, model_plugin_paths))

    if teacher_type is not None:
        if model_plugin_args is None:
            model_plugin_args = {}

        model_plugin_args['model_type'] = 'teacher'
        teacher = create_model(teacher_type, teacher_path, plugin_paths=model_plugin_paths,
                               plugin_args=model_plugin_args, pretrained=teacher_pretrained, num_classes=num_classes)
        kd_settings = KDSettings(teacher, kd_temp_student, kd_temp_teacher,
                                 kd_weight, kd_contradict_hinton)
        print('Created teacher model of type {} with num_classes:{}, pretrained:{} / model_path:{}'
              .format(teacher_type, num_classes, teacher_pretrained, teacher_path))
        print('Created kd settings with teacher model of temp_student:{}, temp_teacher:{}, weight:{}, con_hinton:{}'
              .format(kd_temp_student, kd_temp_teacher, kd_weight, kd_contradict_hinton))
    else:
        teacher = None
        kd_settings = None
        print('Not using knowledge distillation')

    model, device, device_ids = model_to_device(model, device_desc)

    if teacher is not None:
        teacher, _, __ = model_to_device(teacher, device_desc)

    print('Transferred model(s) to device {} with ids {}'.format(device, device_ids))

    loss = CrossEntropyLossWrapper(
        extras={'top1acc': TopKAccuracy(1), 'top5acc': TopKAccuracy(5)}, kd_settings=kd_settings
    )
    print('Created loss {}'.format(loss))

    ####################################################################################################################
    #
    # Optimizer and modifiers (manager and scheduled optimizer) setup
    #
    ####################################################################################################################

    optimizer = SGD(model.parameters(), init_lr, momentum, dampening, weight_decay, nesterov)

    if load_optim:
        epoch = load_optimizer(model_path, optimizer)
    else:
        epoch = -1

    print('Created optimizer with init_lr:{} momentum:{} dampening:{} weight_decay:{} nesterov:{}'
          .format(init_lr, momentum, dampening, weight_decay, nesterov))

    schedule_path = os.path.abspath(os.path.expanduser(schedule_path))
    modifier_manager = ScheduledModifierManager.from_yaml(schedule_path)
    optimizer = ScheduledOptimizer(optimizer, model, modifier_manager,
                                   steps_per_epoch=math.ceil(len(train_dataset) / train_batch_size))
    print('Created modifier manager and optimizer from {}'.format(schedule_path))

    ####################################################################################################################
    #
    # Logging and tensorboard writer setup
    #
    ####################################################################################################################

    save_dir = os.path.abspath(os.path.expanduser(save_dir))
    model_tag = '{}_{}-{}-{}'.format(model_type.replace('/', '.'),
                                     teacher_type.replace('/', '.') if teacher_type else 'None', dataset_type,
                                     os.path.basename(schedule_path).split('.')[0])
    model_id = model_tag
    model_inc = 0
    logs_dir = os.path.abspath(os.path.expanduser(os.path.join(save_dir, 'tensorboard-logs')))

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    while os.path.exists(os.path.join(logs_dir, model_id)):
        model_inc += 1
        model_id = '{}__{:02d}'.format(model_tag, model_inc)

    print('Model id is set to {}'.format(model_id))
    logs_dir = os.path.join(logs_dir, model_id)
    os.makedirs(logs_dir)

    writer = SummaryWriter(logdir=logs_dir, comment='modifier training')
    print('Created summary writer logging to \n{}'.format(logs_dir))

    ####################################################################################################################
    #
    # Training and testing setup
    #
    ####################################################################################################################

    def _train_loss_callback(_epoch: int, _step: int, _batch_size: int,
                             _data: Iterable[Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]]],
                             _pred: Union[Tensor, Iterable[Tensor]], _losses: Dict[str, Tensor]):
        _losses[DEFAULT_LOSS_KEY] = optimizer.loss_update(_losses[DEFAULT_LOSS_KEY])

    def _train_batch_end_callback(_epoch: int, _step: int, _batch_size: int,
                                  _data: Iterable[Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]]],
                                  _pred: Union[Tensor, Iterable[Tensor]], _losses: Dict[str, Tensor]):
        _step = _epoch * len(train_dataset) + _step

        for _loss, _value in _losses.items():
            writer.add_scalar('Train/{}'.format(_loss), _value, _step)

        writer.add_scalar('Train/Learning Rate', optimizer.learning_rate, _step)
        writer.add_scalar('Train/Batch Size', _batch_size, _step)
        writer.add_scalar('Train/Epoch', _epoch, _step)

        if track_gradients:
            for name, mod in model.named_modules():
                for param_name, param in mod.named_parameters():
                    if not param.requires_grad:
                        continue

                    grad = param.grad.view(-1)

                    if 'norm' in track_gradients:
                        writer.add_scalar('Gradients/{}/{}/norm'.format(name, param_name), grad.abs().sum(), _step)

                    if 'mean' in track_gradients:
                        writer.add_scalar('Gradients/{}/{}/mean'.format(name, param_name), grad.mean(), _step)

                    if 'var' in track_gradients:
                        writer.add_scalar('Gradients/{}/{}/var'.format(name, param_name), grad.var(), _step)

                    if 'hist' in track_gradients:
                        writer.add_histogram('Gradients/{}/{}/hist'.format(name, param_name), grad, _step)

    trainer = ModuleTrainer(model, device, loss, optimizer)
    trainer.run_hooks.register_batch_loss_hook(_train_loss_callback)
    trainer.run_hooks.register_batch_end_hook(_train_batch_end_callback)
    tester = ModuleTester(model, device, loss)

    ####################################################################################################################
    #
    # Kernel Sparsity and Activation Sparsity tracking setup
    #
    ####################################################################################################################

    analyze_layers = [key for key, mod in model.named_modules() if isinstance(mod, _ConvNd) or isinstance(mod, Linear)]
    ks_analyzers = ModuleKSAnalyzer.analyze_layers(model, analyze_layers)  # type: List[ModuleKSAnalyzer]
    as_analyzers = ModuleASAnalyzer.analyze_layers(model, analyze_layers, division=None,
                                                   track_inputs_sparsity=True)  # type: List[ModuleASAnalyzer]
    as_dataset = EarlyStopDataset(train_test_dataset, early_stop=1000 if len(train_test_dataset) > 1000 else -1)

    ####################################################################################################################
    #
    # Convenience function for testing validation and training dataset as well as tracking KS and AS
    #
    ####################################################################################################################

    def _run_model_tests(_epoch: int):
        print('Testing validation dataset for epoch {}'.format(_epoch))
        _val_results = tester.run_epoch(_create_data_loader(val_dataset, False), _epoch)

        for _loss in _val_results.results.keys():
            writer.add_scalar('Test/Validation/{}'.format(_loss), _val_results.result_mean(_loss), _epoch)
            print('Epoch {} Test/Validation/{}: {}'.format(_epoch, _loss, _val_results.result_mean(_loss)))

        print('Testing training dataset for epoch {}'.format(_epoch))
        _train_results = tester.run_epoch(_create_data_loader(train_test_dataset, False), _epoch)

        for _loss in _train_results.results.keys():
            writer.add_scalar('Test/Training/{}'.format(_loss), _train_results.result_mean(_loss), _epoch)
            print('Epoch {} Test/Training/{}: {}'.format(_epoch, _loss, _train_results.result_mean(_loss)))

        if track_as:
            print('Testing activation sparsity for training dataset for epoch {}'
                  .format(_epoch))

            for _analyzer in as_analyzers:
                _analyzer.clear()
                _analyzer.enable()

            tester.run_epoch(_create_data_loader(as_dataset, False), _epoch)

            for _index, _analyzer in enumerate(as_analyzers):
                _analyzer.disable()
                writer.add_scalar('Act Sparsity/{}'.format(analyze_layers[_index].replace('module.', '')),
                                  _analyzer.inputs_sparsity_mean, epoch)

        if track_ks:
            for _ks_layer in ks_analyzers:
                writer.add_scalar('Kernel Sparsity/{}'.format(_ks_layer.name.replace('module.', '')),
                                  _ks_layer.param_sparsity, epoch)

    ####################################################################################################################
    #
    # Training section
    #
    ####################################################################################################################

    if not save_epochs:
        save_epochs = []

    print('Running baseline tests')
    _run_model_tests(epoch)

    while epoch < modifier_manager.max_epochs - 1:
        epoch += 1
        print('Starting epoch {}'.format(epoch))
        optimizer.epoch_start()
        trainer.run_epoch(_create_data_loader(train_dataset, True), epoch)

        print('Completed training for epoch {}'.format(epoch))
        print('Running baseline tests')
        _run_model_tests(epoch)

        if ((epoch in save_epochs) or
                (epoch >= save_after_epoch and save_epoch_mod > 0 and epoch % save_epoch_mod == 0)):
            save_path = os.path.join(save_dir, '{}.checkpoint-{:03d}.pth'.format(model_id, epoch))
            print('Saving checkpoint to {}'.format(save_path))
            save_model(save_path, model, optimizer, epoch)

        optimizer.epoch_end()

    save_path = os.path.join(save_dir, '{}.pth'.format(model_id))
    print('Saving result to {}'.format(save_path))
    save_model(save_path, model, optimizer, epoch)
    print('Completed')

    # add sleep to make sure all background processes have finished, ex tensorboard writing
    time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description='Train a model for a given dataset with a given schedule')

    # schedule device and model arguments
    parser.add_argument('--schedule-path', type=str, required=True,
                        help='The path to the yaml file containing the modifiers and schedule to apply them with')
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

    # teacher and knowledge distillation settings
    parser.add_argument('--teacher-type', type=str, default=None,
                        help='The model type to load for a knowledge distillation teacher, default is not to load one, '
                             'must be set to enable knowledge distillation')
    parser.add_argument('--teacher-pretrained', type=str, default=None,
                        help='The type of pretrained weights to use for the teacher, default is not to load any, '
                             'ex: imagenet/dense, imagenet/sparse, imagenette/dense')
    parser.add_argument('--teacher-path', type=str, default=None,
                        help='The path to the teachers model file to load a previous state dict from')
    parser.add_argument('--kd-temp-student', type=float, default=5.0,
                        help='The temperature to use for the students predictions')
    parser.add_argument('--kd-temp-teacher', type=float, default=5.0,
                        help='The temperature to use for the teachers predictions')
    parser.add_argument('--kd-weight', type=float, default=0.5,
                        help='The weight to use for the knowledge distillation loss when combining with original loss')
    parser.add_argument('--kd-contradict-hinton', type=bool, default=True,
                        help='in hintons original paper they included temperature^2 as a scaling factor, '
                             'most implementations dropped this factor so contradicting hinton does not scale by it')

    # dataset settings
    parser.add_argument('--dataset-type', type=str, required=True,
                        help='The dataset type to load for training, ex: imagenet, imagenette, cifar10, etc')
    parser.add_argument('--dataset-root', type=str, required=True,
                        help='The root path to where the dataset is stored')
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
    parser.add_argument('--init-lr', type=float, default=10e-9,
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
    parser.add_argument('--save-dir', type=str, default='modifiers-training',
                        help='The path to the directory for saving results')
    parser.add_argument('--save-after-epoch', type=int, default=-1,
                        help='the epoch after which to start saving checkpoints with either '
                             'save-epochs or save-epoch-mod args')
    parser.add_argument('--save-epochs', type=int, default=[], nargs='+',
                        help='comma separated list of epochs to save checkpoints at')
    parser.add_argument('--save-epoch-mod', type=int, default=-1,
                        help='the modulus to save checkpoints at, '
                             'ie 2 will save every other epoch starting with an even number')

    # additional options
    parser.add_argument('--track-ks', type=bool, default=False,
                        help='Track the sparsity of kernels for all convolutional / linear layers')
    parser.add_argument('--track-as', type=bool, default=False,
                        help='Track the sparsity of the inputs for all convolutional / linear layers')
    parser.add_argument('--track-grads', default=None, nargs='+',
                        help='Track the gradients of trainable params in tensorboard, '
                             'multiple ways to track so multiple args can be supplied '
                             'options and/or: norm, mean, var, hist')

    # plugin options
    parser.add_argument('--model-plugin-paths', default=None, nargs='+',
                        help='plugins to load for handling a model and possibly teacher after creation')
    parser.add_argument('--model-plugin-args', type=json.loads, default={},
                        help='json string containing the args to pass to the model plugins when executing')

    # debug options
    parser.add_argument('--debug-early-stop', type=int, default=-1,
                        help='Early stop going through the datasets to debug issues, '
                             'will create the datasets with this number of items')

    args = parser.parse_args()
    train_modifiers_schedule(
        args.schedule_path, args.device,
        args.model_type, args.pretrained, args.model_path,
        args.teacher_type, args.teacher_pretrained, args.teacher_path,
        args.kd_temp_student, args.kd_temp_teacher, args.kd_weight, convert_to_bool(args.kd_contradict_hinton),
        args.dataset_type, args.dataset_root,
        args.train_batch_size, args.test_batch_size, args.num_workers, convert_to_bool(args.pin_memory),
        convert_to_bool(args.load_optim), args.init_lr, args.momentum, args.dampening, args.weight_decay, convert_to_bool(args.nesterov),
        args.save_dir, args.save_after_epoch, args.save_epochs, args.save_epoch_mod,
        convert_to_bool(args.track_ks), convert_to_bool(args.track_as), args.track_grads,
        args.model_plugin_paths, args.model_plugin_args,
        args.debug_early_stop
    )


if __name__ == '__main__':
    main()
