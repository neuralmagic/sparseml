from typing import Tuple, List, Callable, Dict, Any, Union
import json
from tqdm import auto

import torch
from torch.nn import Module, DataParallel
from torch.utils.data import Dataset, DataLoader

from ...datasets import EarlyStopDataset, CacheableDataset
from ...models import model_to_device
from ...utils import ModuleTester, clean_path, create_parent_dirs, DEFAULT_LOSS_KEY, ModuleRunFuncs
from ..module_analyzer import ModuleAnalyzer
from ..helpers import get_conv_layers, get_linear_layers
from .mask import KSLayerParamMask


__all__ = ['one_shot_sensitivity_analysis', 'OneShotLayerSensitivity',
           'save_one_shot_sensitivity_analysis']


class OneShotLayerSensitivity(object):
    def __init__(self, name: str, type_: str, execution_order: int = -1, measured: List[Tuple[float, float]] = None):
        self.name = name
        self.type_ = type_
        self.execution_order = execution_order
        self.measured = measured

    def __repr__(self):
        return 'OneShotLayerSensitivity({})'.format(self.json())

    @property
    def integrate(self) -> float:
        total = torch.tensor(0.0)

        for index, (sparsity, loss) in enumerate(self.measured):
            prev_sparsity = self.measured[index - 1][0] if index > 0 else 0.0
            next_sparsity = self.measured[index + 1][0] if index < len(self.measured) - 1 else 1.0
            x_dist = (next_sparsity - sparsity) / 2.0 + (sparsity - prev_sparsity) / 2.0
            total += x_dist * loss

        return total.item()

    def dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type_,
            'measured': [{'sparsity': val[0], 'loss': val[1]} for val in self.measured],
            'integral_loss': self.integrate
        }

    def json(self) -> str:
        return json.dumps(self.dict())


def one_shot_sensitivity_analysis(
        model: Module, data: Dataset, loss_fn: Callable, device: str, batch_size: int, samples_per_check: int,
        sparsity_levels: List[int] = None, cache_data: bool = True,
        loader_args: Dict = None, data_loader_const: Callable = DataLoader, tester_run_funcs: ModuleRunFuncs = None,
        progress_callback: Callable[[int, int, str, List[int], int], None] = None
) -> List[OneShotLayerSensitivity]:
    if len(data) > samples_per_check > 0:
        data = EarlyStopDataset(data, samples_per_check)

    if loader_args is None:
        loader_args = {}

    if cache_data:
        # cacheable dataset does not work with parallel data loaders
        if 'num_workers' in loader_args and loader_args['num_workers'] != 0:
            raise ValueError('num_workers must be 0 for dataset cache')

        loader_args['num_workers'] = 0
        data = CacheableDataset(data)

    if sparsity_levels is None:
        sparsity_levels = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.9999]

    layers = {}
    layers.update(get_conv_layers(model))
    layers.update(get_linear_layers(model))
    progress = auto.tqdm(total=len(layers) * len(sparsity_levels) * samples_per_check,
                         desc='Sensitivity Analysis')

    orig_device = next(model.parameters()).device

    analyzer = ModuleAnalyzer(model, enabled=True)
    model, device, device_ids = model_to_device(model, device)
    device_str = device if device_ids is None or len(device_ids) < 2 else '{}:{}'.format(device, device_ids[0])

    def _batch_end(_epoch: int, _step: int, _batch_size: int, _data: Any, _pred: Any, _losses: Any):
        analyzer.enabled = False
        progress.update(_batch_size)

    tester = ModuleTester(model, device_str, loss_fn)
    batch_end_hook = tester.run_hooks.register_batch_end_hook(_batch_end)

    if tester_run_funcs is not None:
        tester.run_funcs.copy(tester_run_funcs)

    sensitivities = []

    for layer_index, (name, layer) in enumerate(layers.items()):
        if progress_callback is not None:
            progress_callback(layer_index, len(layers), name, sparsity_levels, -1)

        sparsities_loss = []
        mask = KSLayerParamMask(layer, store_init=True, store_unmasked=False, track_grad_mom=-1)
        mask.enabled = True

        for sparsity_index, sparsity_level in enumerate(sparsity_levels):
            mask.set_param_mask_from_sparsity(sparsity_level)
            data_loader = data_loader_const(data, batch_size, **loader_args)
            res = tester.run(data_loader, desc='', show_progress=False, track_results=True)
            sparsities_loss.append((sparsity_level, res.result_mean(DEFAULT_LOSS_KEY).item()))

            if progress_callback is not None:
                progress_callback(layer_index, len(layers), name, sparsity_levels, sparsity_index)

        mask.enabled = False
        mask.reset()
        del mask

        desc = analyzer.layer_desc(name)
        sensitivities.append(OneShotLayerSensitivity(name, desc.type_, desc.execution_order, sparsities_loss))

        if progress_callback is not None:
            progress_callback(layer_index, len(layers), name, sparsity_levels, len(sparsity_levels) + 1)

    progress.close()
    sensitivities.sort(key=lambda val: val.execution_order)
    batch_end_hook.remove()

    if isinstance(model, DataParallel):
        model = model.module

    model.to(orig_device)

    return sensitivities


def save_one_shot_sensitivity_analysis(layer_sensitivities: List[OneShotLayerSensitivity], path: str):
    path = clean_path(path)
    create_parent_dirs(path)
    sens_object = {
        'layer_sensitivities': [sens.dict() for sens in layer_sensitivities]
    }

    with open(path, 'w') as file:
        json.dump(sens_object, file)

