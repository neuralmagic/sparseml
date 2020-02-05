from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from ...models import model_to_device
from ...utils import LossWrapper, ModuleTester, ModuleRunResults
from ..helpers import get_layer
from .fatrelu import convert_relus_to_fat, FATReLU
from .analyzer import ModuleASAnalyzer


__all__ = ['LayerBoostResults', 'ModuleASOneShootBooster']


class LayerBoostResults(object):
    def __init__(self, name: str, threshold: float,
                 boosted_as: Tensor, boosted_loss: ModuleRunResults,
                 baseline_as: Tensor, baseline_loss: ModuleRunResults):
        self._name = name
        self._threshold = threshold
        self._boosted_as = boosted_as
        self._boosted_loss = boosted_loss
        self._baseline_as = baseline_as
        self._baseline_loss = baseline_loss

    @property
    def name(self):
        return self._name

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def boosted_as(self) -> Tensor:
        return self._boosted_as

    @property
    def boosted_loss(self) -> ModuleRunResults:
        return self._boosted_loss

    @property
    def baseline_as(self) -> Tensor:
        return self._baseline_as

    @property
    def baseline_loss(self) -> ModuleRunResults:
        return self._baseline_loss


class ModuleASOneShootBooster(object):
    def __init__(self, module: Module, device: str, dataset: Dataset, batch_size: int, loss: LossWrapper,
                 data_loader_kwargs: Dict):
        self._module = module
        self._device = device
        self._dataset = dataset
        self._batch_size = batch_size
        self._loss = loss
        self._dataloader_kwargs = data_loader_kwargs if data_loader_kwargs else {}

    def run_layers(self, layers: List[str], max_target_metric_loss: float, metric_key: str, metric_increases: bool,
                   precision: float = 0.001) -> Dict[str, LayerBoostResults]:
        fat_relus = convert_relus_to_fat(self._module, inplace=True)  # type: Dict[str, FATReLU]

        for layer in layers:
            if layer not in fat_relus:
                raise KeyError('layer {} was specified in the config but is not a boostable layer in the module '
                               '(ie not a relu)'.format(layer))

        module, device, device_ids = model_to_device(self._module, self._device)
        results = {}
        min_thresh = 0.0
        max_thresh = 1.0

        baseline_res, _ = self._measure_layer(None, module, device, 'baseline loss run')

        for layer in layers:
            results[layer] = self._binary_search_fat(layer, module, device, min_thresh, max_thresh,
                                                     max_target_metric_loss, metric_key, metric_increases, precision)

        boosted_res, _ = self._measure_layer(None, module, device, 'boosted loss run')
        results['__module__'] = LayerBoostResults(
            '__module__', -1.0, torch.tensor(-1.0), boosted_res, torch.tensor(-1.0), baseline_res
        )

        return results

    def _binary_search_fat(self, layer: str, module: Module, device: str,
                           min_thresh: float, max_thresh: float,
                           max_target_metric_loss: float, metric_key: str, metric_increases: bool,
                           precision: float = 0.001):
        print('\n\n\nstarting binary search for layer {} between ({}, {})...'.format(layer, min_thresh, max_thresh))
        base_res, base_as = self._measure_layer(layer, module, device,
                                                'baseline for layer: {}'.format(layer))

        fat_relu = get_layer(layer, module)  # type: FATReLU
        init_thresh = fat_relu.get_threshold()

        if min_thresh > init_thresh:
            min_thresh = init_thresh

        while True:
            thresh = ModuleASOneShootBooster._get_mid_point(min_thresh, max_thresh)
            fat_relu.set_threshold(thresh)

            thresh_res, thresh_as = self._measure_layer(layer, module, device,
                                                        'threshold for layer: {} @ {:.4f} ({:.4f}, {:.4f})'
                                                        .format(layer, thresh, min_thresh, max_thresh))

            if ModuleASOneShootBooster._passes_loss(base_res, thresh_res,
                                                    max_target_metric_loss, metric_key, metric_increases):
                min_thresh = thresh
                print('loss check passed for max change: {:.4f}'.format(max_target_metric_loss))
                print('   current loss: {:.4f} baseline loss: {:.4f}'
                      .format(thresh_res.result_mean(metric_key), base_res.result_mean(metric_key)))
                print('   current AS: {:.4f} baseline AS: {:.4f}'.format(thresh_as, base_as))
            else:
                max_thresh = thresh
                print('loss check failed for max change: {:.4f}'.format(max_target_metric_loss))
                print('   current loss: {:.4f} baseline loss: {:.4f}'
                      .format(thresh_res.result_mean(metric_key), base_res.result_mean(metric_key)))
                print('   current AS: {:.4f} baseline AS: {:.4f}'.format(thresh_as, base_as))

            if max_thresh - min_thresh <= precision:
                break

        print('completed binary search for layer {}')
        print('   found threshold: {}'.format(thresh))
        print('   AS delta: {:.4f} ({:.4f} => {:.4f})'.format(thresh_as - base_as, base_as, thresh_as))

        return LayerBoostResults(layer, thresh, thresh_as, thresh_res, base_as, base_res)

    def _measure_layer(self, layer: Union[str, None], module: Module,
                       device: str, desc: str) -> Tuple[ModuleRunResults, Tensor]:
        layer = get_layer(layer, module) if layer else None
        as_analyzer = None

        if layer:
            as_analyzer = ModuleASAnalyzer(layer, division=None, track_outputs_sparsity=True)
            as_analyzer.enable()

        tester = ModuleTester(module, device, self._loss)
        data_loader = DataLoader(self._dataset, self._batch_size, **self._dataloader_kwargs)
        results = tester.run(data_loader, desc=desc, show_progress=True, track_results=True)

        if as_analyzer:
            as_analyzer.disable()

        return results, as_analyzer.outputs_sparsity_mean if as_analyzer else None

    @staticmethod
    def _get_mid_point(start: float, end: float) -> float:
        return (end - start) / 2.0 + start

    @staticmethod
    def _passes_loss(base_res: ModuleRunResults, thresh_res: ModuleRunResults,
                     max_target_metric_loss: float, metric_key: str, metric_increases: bool) -> bool:
        diff = thresh_res.result_mean(metric_key) - base_res.result_mean(metric_key) if metric_increases else \
            base_res.result_mean(metric_key) - thresh_res.result_mean(metric_key)

        return diff < max_target_metric_loss
