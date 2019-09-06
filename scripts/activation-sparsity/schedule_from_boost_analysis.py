from typing import List
import argparse
import glob
import os
import json
import numpy
from tqdm import tqdm


def schedule_from_boost_analysis(analysis_dir: str, losses_per_layer: List[str], min_sparsity: float,
                                 train_epochs: int, init_lr: float, gamma: float, lr_milestones: List[int]):

    output = ('version: 1'
              '\n'
              '\nmodifiers:'
              '\n'
              '\n  - !LearningRateModifier'
              '\n    start_epoch: 0.0'
              '\n    end_epoch: {}'
              '\n    update_frequency: 1.0'
              '\n    lr_class: MultiStepLR'
              '\n    lr_kwargs:'
              '\n      milestones: [{}]'
              '\n      gamma: {}'
              '\n    init_lr: {}'
              '\n'.format(train_epochs,
                          ','.join([str(mil) for mil in lr_milestones]) if len(lr_milestones) > 0 else train_epochs + 1,
                          gamma if len(lr_milestones) > 0 else 1.0,
                          init_lr))

    losses_per_layer = {loss.split('=')[0]: float(loss.split('=')[1]) for loss in losses_per_layer}
    loss_keys = [loss for loss in losses_per_layer.keys()]
    paths = [path for path in glob.glob(os.path.join(analysis_dir, '*.json'))]
    paths.sort(key=lambda x: os.path.getmtime(x))

    for layer_path in tqdm(paths):
        layer_name = os.path.basename(layer_path).replace('.json', '')

        with open(layer_path, 'r') as layer_file:
            layer = json.load(layer_file)

        layer_vals = []

        for thresh, vals in layer['thresholds'].items():
            layer_vals.append({
                'losses': {key: numpy.mean(vals['losses'][key]) for key in loss_keys},
                'threshold': float(thresh),
                'sparsity': numpy.mean(vals['sparsities'])
            })

        baseline = layer_vals[0]
        prev_val = baseline
        layer_vals = layer_vals[1:]

        for layer_val in layer_vals:
            limit_thresh = None

            for loss_key in loss_keys:
                if baseline['losses'][loss_key] - layer_val['losses'][loss_key] >= losses_per_layer[loss_key]:
                    calc_thresh = numpy.interp(baseline['losses'][loss_key] - losses_per_layer[loss_key],
                                               [layer_val['losses'][loss_key], prev_val['losses'][loss_key]],
                                               [layer_val['threshold'], prev_val['threshold']])
                    calc_sparsity = numpy.interp(baseline['losses'][loss_key] - losses_per_layer[loss_key],
                                                 [layer_val['losses'][loss_key], prev_val['losses'][loss_key]],
                                                 [layer_val['sparsity'], prev_val['sparsity']])

                    if limit_thresh is None or calc_thresh < limit_thresh:
                        limit_thresh = calc_thresh if calc_sparsity >= min_sparsity else -1

            prev_val = layer_val

            if limit_thresh is None:
                continue

            if limit_thresh > 0.0:
                # valid threshold, add to output
                output += ('\n'
                           '\n  - !SetParamModifier'
                           '\n      start_epoch: 0.0'
                           '\n      param: threshold'
                           '\n      val: {}'
                           '\n      layers:'
                           '\n        - {}'.format(limit_thresh, layer_name))

            break

    print('\n')
    print('\n')
    print('\n')
    output += '\n'
    print(output)


def main():
    parser = argparse.ArgumentParser(description='Create a modifier schedule for boosting from boost analysis')

    # schedule device and model arguments
    parser.add_argument('--analysis-dir', type=str, required=True,
                        help='The location of directory containing the analysis files')
    parser.add_argument('--losses-per-layer', required=True, nargs='+',
                        help='the loss names and the value that can be lost per layer to grab the proper threshold '
                             'ex: loss=0.01 top1acc=0.1')
    parser.add_argument('--min-sparsity', type=float, default=0.4,
                        help='The minimum sparsity needed after applying the boost')
    parser.add_argument('--train-epochs', type=int, default=3,
                        help='The number of epochs to train for')
    parser.add_argument('--init-lr', type=float, default=0.005,
                        help='the initial learning rate to use')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='the gamma multiplier for each milestone')
    parser.add_argument('--lr-milestones', default=[1, 2], nargs='+',
                        help='the epochs to lessen the learning rate at by multiplying by gamma')

    args = parser.parse_args()
    schedule_from_boost_analysis(
        args.analysis_dir, args.losses_per_layer, args.min_sparsity,
        args.train_epochs, args.init_lr, args.gamma, args.lr_milestones
    )


if __name__ == '__main__':
    main()
