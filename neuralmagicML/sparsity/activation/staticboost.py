from typing import Union, Dict, List, Tuple
from tqdm import tqdm
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from ...models import model_to_device
from ...datasets import EarlyStopDataset
from ...utils import LossWrapper, ModuleTester, ModuleTestResults
from .fatrelu import convert_relus_to_fat, FATReLU
from .analyzer import ASAnalyzerModule, ASAnalyzerLayer


__all__ = ['StaticBoosterResults', 'ModuleStaticBooster']


class StaticBoosterResults(object):
    def __init__(self, baseline_losses: ModuleTestResults, final_losses: ModuleTestResults,
                 layers: List[str], baseline_analyzers: List[ASAnalyzerLayer], final_analyzers: List[ASAnalyzerLayer]):
        self._baseline_losses = baseline_losses
        self._final_losses = final_losses
        self._layers_sparsity = {}

        for layer, baseline, final in zip(layers, baseline_analyzers, final_analyzers):
            self._layers_sparsity[layer] = (baseline, final)

    @property
    def baseline_losses(self) -> ModuleTestResults:
        return self._baseline_losses

    @property
    def final_losses(self) -> ModuleTestResults:
        return self._final_losses

    @property
    def layers_sparsity(self) -> Dict[str, Tuple[ASAnalyzerLayer, ASAnalyzerLayer]]:
        return self._layers_sparsity

    def layer_baseline_sparsity(self, layer: str) -> ASAnalyzerLayer:
        return self._layers_sparsity[layer][0]

    def layer_final_sparsity(self, layer: str) -> ASAnalyzerLayer:
        return self._layers_sparsity[layer][1]


class ModuleStaticBooster(object):
    def __init__(self, module: Module, device: str,
                 dataset: Dataset, batch_size: int, sample_size: int, loss: LossWrapper, **kwargs):
        module, device, device_ids = model_to_device(module, device)
        self._module = module
        self._device = device
        self._device_ids = device_ids
        self._dataset = EarlyStopDataset(dataset, early_stop=sample_size)
        self._batch_size = batch_size
        self._loss = loss
        self._dataloader_kwargs = {key.replace('dataloader_', ''): val
                                   for key, val in kwargs.items() if key.startswith('dataloader_')}

    def boost_layers(self, layers: List[str], losses_criteria: Dict[str, Dict[str, float]],
                     min_boosted_sparsity: float = 0.4, precision: float = 0.001):
        for loss_name in losses_criteria.keys():
            if loss_name not in self._loss.available_losses:
                raise KeyError('loss {} was specified in the config but is not available in the loss given: {}'
                               .format(loss_name, self._loss.available_losses))

        fat_relus = convert_relus_to_fat(self._module, inplace=True)  # type: Dict[str, FATReLU]

        for layer in layers:
            if layer not in fat_relus:
                raise KeyError('layer {} was specified in the config but is not a boostable layer in the module '
                               '(ie not a relu)'.format(layer))

        baseline_losses, baseline_analyzers = self._measure_sparse_losses(layers, include_inp_dist=True)

        for layer, analyzer in tqdm(zip(layers, baseline_analyzers), desc='boosting layers', total=len(layers)):
            print('\n\n\n')
            self._boost_layer(layer, fat_relus[layer], losses_criteria,
                              baseline_losses, analyzer, precision, min_boosted_sparsity)

        final_losses, final_analyzers = self._measure_sparse_losses(layers, include_inp_dist=True)

        return StaticBoosterResults(baseline_losses, final_losses, layers, baseline_analyzers, final_analyzers)

    def _boost_layer(self, layer: str, fat_relu: FATReLU, losses_criteria: Dict[str, Dict[str, float]],
                     baseline_losses: ModuleTestResults, baseline_analyzer: ASAnalyzerLayer,
                     precision: float, min_boosted_sparsity: float):
        print('')
        print('boosting layer {}: checking new baseline loss'.format(layer))
        baseline_layer_losses, _ = self._measure_sparse_losses(None)

        org_thresh = fat_relu.get_threshold()
        min_thresh = org_thresh
        max_thresh = baseline_analyzer.inputs_sample_max.item()

        if max_thresh - min_thresh <= precision:
            max_thresh += 1.5 * precision  # shouldn't hit this, but make sure we always run at least once

        analyzers = None

        while max_thresh - min_thresh > precision:
            check_thresh = (max_thresh - min_thresh) / 2.0
            print('')
            print('boosting layer {}: checking for threshold {:.4f}'.format(layer, check_thresh))
            fat_relu.set_threshold(check_thresh)
            losses, analyzers = self._measure_sparse_losses([layer])

            valid = ModuleStaticBooster._check_loss_criteria(losses, baseline_layer_losses, baseline_losses,
                                                             losses_criteria)

            if not valid:
                print('')
                print('boosting layer {}: threshold {:.4f} did not pass, decreasing'.format(layer, check_thresh))
                max_thresh = check_thresh
            else:
                print('')
                print('boosting layer {}: threshold {:.4f} passed, increasing'.format(layer, check_thresh))
                min_thresh = check_thresh

        analyzer = analyzers[0]

        if analyzer.outputs_sparsity_mean < min_boosted_sparsity:
            print('')
            print('boosting layer {}: finished but sparsity of {:.4f} not greater than the min {.4f}, reverting'
                  .format(layer, analyzer.outputs_sparsity_mean, min_boosted_sparsity))
            fat_relu.set_threshold(org_thresh)
        else:
            print('')
            print('boosting layer {}: finished with sparsity of {:.4f} and increased from {:.4f}'
                  .format(layer, analyzer.outputs_sparsity_mean, baseline_analyzer.outputs_sparsity_mean))

    def _measure_sparse_losses(self, layers: Union[List[str], None],
                               include_inp_dist: bool = False) -> Tuple[ModuleTestResults, List[ASAnalyzerLayer]]:
        analyzer_layers = [ASAnalyzerLayer(layer, division=0, track_outputs_sparsity=True,
                                           inputs_sample_size=100 if include_inp_dist else 0)
                           for layer in layers] if layers is not None else []
        analyzer = ASAnalyzerModule(self._module, analyzer_layers)
        tester = ModuleTester(self._module, self._device, self._loss)
        data_loader = DataLoader(self._dataset, self._batch_size, **self._dataloader_kwargs)

        analyzer.enable_layers()
        results = tester.test_epoch(data_loader, -1)
        analyzer.disable_layers()

        return results, analyzer_layers

    @staticmethod
    def _check_loss_criteria(layer: ModuleTestResults, layer_baseline: ModuleTestResults,
                             baseline: ModuleTestResults, losses_criteria: Dict[str, Dict[str, float]]) -> bool:
        for loss, criteria in losses_criteria.items():
            boosted_loss = layer.result_mean(loss)
            baseline_layer_loss = layer_baseline.result_mean(loss)
            baseline_loss = baseline.result_mean(loss)

            if baseline_layer_loss - boosted_loss > criteria['max_layer_loss']:
                return False

            if baseline_loss - boosted_loss > criteria['max_total_loss']:
                return False

        return True
