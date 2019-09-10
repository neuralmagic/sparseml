from typing import Union, List, Tuple, Dict
import copy
import math
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from ...models import model_to_device
from ...datasets import EarlyStopDataset
from ...utils import LossWrapper, ModuleTester, ModuleTestResults, ModuleTrainer
from ..optimizer import ScheduledOptimizer
from ..modifier import ScheduledModifierManager
from .fatrelu import set_relu_to_fat, FATReLU
from .analyzer import ASAnalyzerModule, ASAnalyzerLayer


__all__ = ['ModuleASBoostAnalysis']


class ModuleASBoostAnalysis(object):
    def __init__(self, module: Module, device: str, loss: LossWrapper,
                 test_dataset: Dataset, test_batch_size: int, sample_size: int,
                 train_dataset: Union[Dataset, None] = None, train_batch_size: int = 64,
                 train_size: Union[float, int] = 0.2, train_schedule_path: Union[str, None] = None, **kwargs):
        self._module = module.cpu()
        self._device = device
        self._loss = loss
        self._test_dataset = EarlyStopDataset(test_dataset, early_stop=sample_size)
        self._test_batch_size = test_batch_size

        if train_dataset:
            train_early_stop = round(train_size) if train_size > 1.0 else round(train_size * len(train_dataset))
            self._train_epochs = math.ceil(train_early_stop / len(train_dataset))
            self._train_dataset = EarlyStopDataset(train_dataset, early_stop=round(train_early_stop)) \
                if train_early_stop < len(train_dataset) else train_dataset
        else:
            self._train_epochs = 0
            self._train_dataset = None

        self._train_batch_size = train_batch_size
        self._train_schedule_path = train_schedule_path
        self._dataloader_kwargs = {key.replace('dataloader_', ''): val
                                   for key, val in kwargs.items() if key.startswith('dataloader_')}
        self._optim_kwargs = {key.replace('optim_', ''): val
                              for key, val in kwargs.items() if key.startswith('optim_')}

    def analyze_layer(self, layer: str, init_thresh: float = 0.001,
                      final_thresh_per: float = 0.99, final_thresh_val: Union[None, float] = None,
                      thresh_mult: float = 2.0) -> List[Tuple[float, ModuleTestResults, ASAnalyzerLayer]]:
        layer_results = []

        fat = set_relu_to_fat(self._module, layer, inplace=True)  # type: FATReLU
        orig_thresh = fat.get_threshold()

        if fat.get_threshold() > 0.0:
            init_thresh += fat.get_threshold()

        print('\n\n\nanalyzing layer {}: checking baseline'.format(layer))
        module, device, device_ids = model_to_device(copy.deepcopy(self._module), self._device)
        baseline_loss, baseline_analyzer = self._measure_sparse_losses(module, device, layer, include_inp_dist=True)
        layer_results.append((fat.get_threshold(), baseline_loss, baseline_analyzer))

        if final_thresh_val is None:
            inputs = torch.cat(baseline_analyzer.inputs_sample)
            top_inputs, _ = torch.topk(inputs.view(-1), int((1.0 - final_thresh_per) * inputs.numel()),
                                       largest=True, sorted=True)
            final_thresh_val = top_inputs[-1].item()

        thresholds = ModuleASBoostAnalysis._thresholds(init_thresh, final_thresh_val, thresh_mult)

        for threshold in tqdm(thresholds, desc='eval thresholds'):
            print('\nanalyzing layer {}: checking threshold {}'.format(layer, threshold))
            fat.set_threshold(threshold)
            module, device, device_ids = model_to_device(copy.deepcopy(self._module), self._device)
            self._train(module, device)
            thresh_loss, thresh_analyzer = self._measure_sparse_losses(module, device, layer)
            layer_results.append((threshold, thresh_loss, thresh_analyzer))

        fat.set_threshold(orig_thresh)

        return layer_results

    def _measure_sparse_losses(self, module: Module, device: str, layer: str,
                               include_inp_dist: bool = False) -> Tuple[ModuleTestResults, ASAnalyzerLayer]:
        analyzer_layers = [ASAnalyzerLayer(layer, division=0, track_outputs_sparsity=True,
                                           inputs_sample_size=100 if include_inp_dist else 0)]
        analyzer = ASAnalyzerModule(module, analyzer_layers)
        tester = ModuleTester(module, device, self._loss)
        data_loader = DataLoader(self._test_dataset, self._test_batch_size, **self._dataloader_kwargs)

        analyzer.enable_layers()
        results = tester.test_epoch(data_loader, -1)
        analyzer.disable_layers()

        return results, analyzer_layers[0]

    def _train(self, module: Module, device: str):
        if self._train_dataset is None:
            return

        optim = SGD(module.parameters(), **self._optim_kwargs)
        manager = [] if self._train_schedule_path is None \
            else ScheduledModifierManager.from_yaml(self._train_schedule_path)
        optim = ScheduledOptimizer(optim, module, manager, math.ceil(len(self._train_dataset) / self._train_batch_size))
        trainer = ModuleTrainer(module, device, self._loss, optim)

        def _train_loss_callback(_epoch: int, _step: int, _x_feature: Tuple[Tensor, ...],
                                 _y_lab: Tensor, _y_pred: Tensor, _losses: Dict[str, Tensor]):
            _losses['loss'] = optim.loss_update(_losses['loss'])

        trainer.register_batch_loss_hook(_train_loss_callback)

        for epoch in range(self._train_epochs):
            data_loader = DataLoader(self._train_dataset, self._train_batch_size, **self._dataloader_kwargs)
            optim.epoch_start()
            trainer.train_epoch(data_loader, epoch)
            optim.epoch_end()

    @staticmethod
    def _thresholds(init_thresh: float, final_thresh: float, mult: float) -> List[float]:
        if final_thresh <= init_thresh:
            return []

        thresholds = [init_thresh]

        while thresholds[-1] < final_thresh:
            thresholds.append(thresholds[-1] * mult)

        return thresholds
