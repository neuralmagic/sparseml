from typing import Union
import argparse
import os
import json
import yaml
import matplotlib.pyplot as plt
import numpy
import pandas
import torch
from torch.nn import Module

from neuralmagicML.datasets import create_dataset, EarlyStopDataset
from neuralmagicML.models import create_model, save_model
from neuralmagicML.sparsity import ModuleStaticBooster, StaticBoosterResults, convert_to_bool
from neuralmagicML.utils import CrossEntropyLossWrapper, TopKAccuracy


def as_staticbooster(config_path: str, device_desc: str,
                     model_type: str, pretrained: Union[bool, str], model_path: Union[None, str],
                     dataset_type: str, dataset_root: str,
                     batch_size: int, sample_size: int, num_workers: int, pin_memory: bool,
                     save_dir: str, debug_early_stop: int):
    ####################################################################################################################
    #
    # Dataset and data loader setup section
    #
    ####################################################################################################################
    dataset, num_classes = create_dataset(dataset_type, dataset_root, train=True, rand_trans=False)
    if debug_early_stop > 0:
        dataset = EarlyStopDataset(dataset, debug_early_stop)
    print('created dataset: \n{}\n'.format(dataset))

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
    # Boosting section
    #
    ####################################################################################################################

    config_path = os.path.abspath(os.path.expanduser(config_path))

    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file)

    booster = ModuleStaticBooster(model, device_desc, dataset, batch_size, sample_size, loss,
                                  dataloader_num_workers=num_workers, dataloader_pin_memory=pin_memory)
    results = booster.boost_layers(config['layers'], config['losses_criteria'])

    ####################################################################################################################
    #
    # Results section
    #
    ####################################################################################################################

    save_dir = os.path.abspath(os.path.expanduser(save_dir))
    model_tag = '{}-{}-{}'.format(model_type.replace('/', '.'), dataset_type,
                                  os.path.basename(config_path).split('.')[0])
    model_id = model_tag
    model_inc = 0

    while os.path.exists(os.path.join(save_dir, model_id)):
        model_inc += 1
        model_id = '{}__{:02d}'.format(model_tag, model_inc)

    print('Model id is set to {}'.format(model_id))
    model_dir = os.path.join(save_dir, model_id)
    os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'boosted.pth')
    save_model(model_path, model)
    print('model saved to {}'.format(model_path))

    _save_as_compare_data(results, model, model_dir)
    _save_input_baseline_data(results, model_dir)
    _save_input_final_data(results, model_dir)

    print('Completed')


def _save_as_compare_data(results: StaticBoosterResults, model: Module, save_dir: str):
    results_txt = _results_text(results)
    print(results_txt)

    with open(os.path.join(save_dir, 'compare.txt'), 'w') as compare_file:
        compare_file.write(results_txt)

    as_baseline = {}
    as_final = {}
    as_full_data = {}
    layer_counter = 0

    for name, _ in model.named_modules():
        if name in results.layers_sparsity:
            plot_name = '{:04d}  {}'.format(layer_counter, name)
            layer_counter += 1
            as_baseline[plot_name] = results.layer_baseline_sparsity(name).outputs_sparsity_mean.item()
            as_final[plot_name] = results.layer_final_sparsity(name).outputs_sparsity_mean.item()
            as_full_data[name] = {
                'baseline': torch.cat(results.layer_baseline_sparsity(name).outputs_sparsity).view(-1).tolist(),
                'final': torch.cat(results.layer_final_sparsity(name).outputs_sparsity).view(-1).tolist()
            }

    as_baseline['__overall__'] = numpy.mean([sparsity for key, sparsity in as_baseline.items()])
    as_final['__overall__'] = numpy.mean([sparsity for key, sparsity in as_final.items()])

    for loss in results.baseline_losses.results.keys():
        as_full_data[loss] = {
            'baseline': torch.cat(results.baseline_losses.results[loss]).view(-1).tolist(),
            'final': torch.cat(results.final_losses.results[loss]).view(-1).tolist()
        }

    height = round(len(as_baseline.keys()) / 4) + 3
    fig = plt.figure(figsize=(10, height))
    ax = fig.add_subplot(111)
    ax.set_title('Boosted Activation Sparsity Comparison')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Layer')
    plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)
    frame = pandas.DataFrame(data={'baseline': as_baseline, 'final': as_final})
    frame.plot.barh(ax=ax)
    plt.savefig(os.path.join(save_dir, 'compare.png'))
    plt.close(fig)

    with open(os.path.join(save_dir, 'compare.json'), 'w') as compare_file:
        json.dump(as_full_data, compare_file)


def _results_text(results: StaticBoosterResults) -> str:
    txt = '\n\n#############################################'
    txt += '\nBoosting Results'
    txt += '\n\nLosses'

    for loss in results.baseline_losses.results.keys():
        txt += '\n    {}: {:.4f}+-{:.4f} -> {:.4f}+-{:.4f}' \
            .format(loss, results.baseline_losses.result_mean(loss).item(),
                    results.baseline_losses.result_std(loss).item(),
                    results.final_losses.result_mean(loss).item(),
                    results.final_losses.result_std(loss).item())

    txt += '\n\nLayers Sparsity'

    for layer in results.layers_sparsity.keys():
        txt += '\n    {}: {:.4f}+-{:.4f} -> {:.4f}+-{:.4f}' \
            .format(layer, results.layer_baseline_sparsity(layer).outputs_sparsity_mean.item(),
                    results.layer_baseline_sparsity(layer).outputs_sparsity_std.item(),
                    results.layer_final_sparsity(layer).outputs_sparsity_mean.item(),
                    results.layer_final_sparsity(layer).outputs_sparsity_std.item())
    txt += '\n\n'

    return txt


def _save_input_baseline_data(results: StaticBoosterResults, save_dir: str):
    # TODO: save baseline input distributions
    pass


def _save_input_final_data(results: StaticBoosterResults, save_dir: str):
    # TODO: save final input distributions
    pass


def main():
    parser = argparse.ArgumentParser(description='Boost the activation sparsity for a model without retraining')

    # schedule device and model arguments
    parser.add_argument('--config-path', type=str, required=True,
                        help='The path to the yaml file containing the config to run '
                             'should contain the layers to boost in the order they should boost '
                             'and the losses criteria for how much of each loss is tolerable before stopping boosting')
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
    parser.add_argument('--batch-size', type=int, required=True,
                        help='The batch size to use while testing')
    parser.add_argument('--sample-size', type=int, required=True,
                        help='The total number of samples to run through for calculating losses and sparsity values')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='The number of workers to use for data loading')
    parser.add_argument('--pin-memory', type=bool, default=True,
                        help='Use pinned memory for data loading')

    # save options
    parser.add_argument('--save-dir', type=str, default='as-staticboosting',
                        help='The path to the directory for saving results')

    # debug options
    parser.add_argument('--debug-early-stop', type=int, default=-1,
                        help='Early stop going through the datasets to debug issues, '
                             'will create the datasets with this number of items')

    args = parser.parse_args()
    as_staticbooster(
        args.config_path, args.device,
        args.model_type, args.pretrained, args.model_path,
        args.dataset_type, args.dataset_root,
        args.batch_size, args.sample_size, args.num_workers, convert_to_bool(args.pin_memory),
        args.save_dir,
        args.debug_early_stop
    )


if __name__ == '__main__':
    main()
