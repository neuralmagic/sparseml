from typing import Union, List, Tuple
import copy
import math
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from ...models import model_to_device
from ...datasets import EarlyStopDataset
from ...utils import LossWrapper, ModuleTester, ModuleTestResults, ModuleTrainer
from .fatrelu import set_relu_to_fat, FATReLU
from .analyzer import ASAnalyzerModule, ASAnalyzerLayer


__all__ = ['ModuleBoostAnalysis']


class ModuleBoostAnalysis(object):
    def __init__(self, module: Module, device: str, loss: LossWrapper,
                 test_dataset: Dataset, test_batch_size: int, sample_size: int,
                 train_dataset: Union[Dataset, None] = None, train_batch_size: int = 64,
                 train_size: Union[float, int] = 0.2, **kwargs):
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
        self._dataloader_kwargs = {key.replace('dataloader_', ''): val
                                   for key, val in kwargs.items() if key.startswith('dataloader_')}
        self._optim_kwargs = {key.replace('optim_', ''): val
                              for key, val in kwargs.items() if key.startswith('optim_')}

    def analyze_layer(self, layer: str, init_thresh: float = 0.001, final_thresh_per: float = 0.95,
                      thresh_mult: float = 2.0) -> List[Tuple[float, ModuleTestResults, ASAnalyzerLayer]]:
        layer_results = []

        fat = set_relu_to_fat(self._module, layer, inplace=True)  # type: FATReLU
        fat.set_threshold(0.0)

        print('\n\n\nanalyzing layer {}: checking baseline'.format(layer))
        module, device, device_ids = model_to_device(copy.deepcopy(self._module), self._device)
        baseline_loss, baseline_analyzer = self._measure_sparse_losses(module, device, layer, include_inp_dist=True)
        layer_results.append((0.0, baseline_loss, baseline_analyzer))

        inputs = torch.cat(baseline_analyzer.inputs_sample)
        top_inputs, _ = torch.topk(inputs.view(-1), int((1.0 - final_thresh_per) * inputs.numel()),
                                   largest=True, sorted=True)
        final_thresh = top_inputs[-1].item()
        thresholds = ModuleBoostAnalysis._thresholds(init_thresh, final_thresh, thresh_mult)

        for threshold in tqdm(thresholds, desc='eval thresholds'):
            print('\nanalyzing layer {}: checking threshold {}'.format(layer, threshold))
            fat.set_threshold(threshold)
            module, device, device_ids = model_to_device(copy.deepcopy(self._module), self._device)
            self._train(module, device)
            thresh_loss, thresh_analyzer = self._measure_sparse_losses(module, device, layer)
            layer_results.append((threshold, thresh_loss, thresh_analyzer))

        fat.set_threshold(0.0)

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
        trainer = ModuleTrainer(module, device, self._loss, optim)

        for epoch in range(self._train_epochs):
            data_loader = DataLoader(self._train_dataset, self._train_batch_size, **self._dataloader_kwargs)
            trainer.train_epoch(data_loader, epoch)

    @staticmethod
    def _thresholds(init_thresh: float, final_thresh: float, mult: float) -> List[float]:
        thresholds = [init_thresh]

        while thresholds[-1] * mult <= final_thresh:
            thresholds.append(thresholds[-1] * mult)

        return thresholds
