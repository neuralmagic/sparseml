"""
code related to calculating kernel sparsity sensitivity analysis for models
"""

from typing import Tuple, List, Callable, Dict, Any, Union
from collections import OrderedDict
import json
import numpy
import matplotlib.pyplot as plt
import pandas

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.utils.hooks import RemovableHandle

from ...datasets import EarlyStopDataset, CacheableDataset
from ...models import model_to_device
from ...utils import (
    ModuleTester,
    LossWrapper,
    clean_path,
    create_parent_dirs,
    DEFAULT_LOSS_KEY,
    ModuleRunFuncs,
    get_conv_layers,
    get_linear_layers,
)
from neuralmagicML.utils.module_analyzer import ModuleAnalyzer
from .mask import ModuleParamKSMask


__all__ = [
    "ModuleParamKSSensitivity",
    "KSSensitivityProgress",
    "ModuleKSSensitivityAnalysis",
]


class ModuleParamKSSensitivity(object):
    """
    Sensitivity results for a module's (layer) param
    """

    def __init__(
        self,
        name: str,
        param_name: str,
        type_: str,
        execution_order: int = -1,
        measured: List[Tuple[float, float]] = None,
    ):
        """
        :param name: name of the module or layer in the parent
        :param param_name: name of the param that was analyzed
        :param type_: type of layer; ex: conv, linear, etc
        :param execution_order: the execution number for this layer in the parent
        :param measured: the measured results, a list of tuples ordered as follows [(sparsity, loss)]
        """
        self.name = name
        self.param_name = param_name
        self.type_ = type_
        self.execution_order = execution_order
        self.measured = measured

    def __repr__(self):
        return "OneShotLayerSensitivity({})".format(self.json())

    @property
    def integral(self) -> float:
        """
        :return: calculate the approximated integral for the sensitivity using the measured results
                 returns the approximated area under the sparsity vs loss curve
        """
        total = torch.tensor(0.0)
        total_dist = 0.0

        for index, (sparsity, loss) in enumerate(self.measured):
            prev_distance = (
                sparsity
                if index == 0
                else (sparsity - self.measured[index - 1][0]) / 2.0
            )
            next_distance = (
                1.0 - sparsity
                if index == len(self.measured) - 1
                else (self.measured[index + 1][0] - sparsity) / 2.0
            )
            x_dist = prev_distance + next_distance
            total_dist += x_dist
            total += x_dist * loss

        return total.item()

    def dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "param_name": self.param_name,
            "type": self.type_,
            "measured": [{"sparsity": val[0], "loss": val[1]} for val in self.measured],
            "integral_loss": self.integral,
        }

    def json(self) -> str:
        return json.dumps(self.dict())


class KSSensitivityProgress(object):
    """
    Simple class for tracking the progress of a sensitivity analysis
    """

    def __init__(
        self,
        layer_index: int,
        layer_name: str,
        layers: List[str],
        sparsity_index: int,
        sparsity_levels: List[float],
        measurement_step: int,
        samples_per_measurement: int,
    ):
        """
        :param layer_index: index of the current layer being evaluated, -1 for None
        :param layer_name: the name of the current layer being evaluated
        :param layers: a list of the layers that are being evaluated
        :param sparsity_index: index of the current sparsity level check for the current layer, -1 for None
        :param sparsity_levels: the sparsity levels to be checked for the current layer
        :param measurement_step: the current number of items processed for the measurement on the layer and sparsity lev
        :param samples_per_measurement: number of items to be processed for each layer and sparsity level
        """
        self.layer_index = layer_index
        self.layer_name = layer_name
        self.layers = layers
        self.sparsity_index = sparsity_index
        self.sparsity_levels = sparsity_levels
        self.measurement_step = measurement_step
        self.samples_per_measurement = samples_per_measurement

    def __repr__(self):
        return (
            "{}(layer_index={}, layer_name={}, layers={}, sparsity_index={}, sparsity_levels={},"
            " measurement_step={}, samples_per_measurement={})".format(
                self.__class__.__name__,
                self.layer_index,
                self.layer_name,
                self.layers,
                self.sparsity_index,
                self.sparsity_levels,
                self.measurement_step,
                self.samples_per_measurement,
            )
        )


class ModuleKSSensitivityAnalysis(object):
    """
    Class for handling running sensitivity analysis for kernel sparsity (model pruning) on modules
    """

    @staticmethod
    def save_sensitivities_json(
        sensitivities: List[ModuleParamKSSensitivity], path: str
    ):
        """
        :param sensitivities: the measured sensitivities
        :param path: the path to save the json file at representing the layer sensitivities
        """
        path = clean_path(path)
        create_parent_dirs(path)
        sens_object = {"ks_sensitivities": [sens.dict() for sens in sensitivities]}

        with open(path, "w") as file:
            json.dump(sens_object, file)

    @staticmethod
    def plot_sensitivities(
        sensitivities: List[ModuleParamKSSensitivity],
        path: Union[str, None],
        normalize: bool = True,
        title: str = None,
    ) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[None, None]]:
        """
        :param sensitivities: the measured sensitivities
        :param path: the path to save an img version of the chart, None to display the plot
        :param normalize: normalize the values to a unit distritibution (0 mean, 1 std)
        :param title: the title to put on the chart
        :return: the created figure and axes if path is None, otherwise (None, None)
        """
        layers, values = zip(*[(layer.name, layer.integral) for layer in sensitivities])

        if normalize:
            mean = numpy.mean(values)
            std = numpy.std(values)
            values = [(val - mean) / std for val in values]

        height = round(len(layers) / 4) + 3
        fig = plt.figure(figsize=(12, height))
        ax = fig.add_subplot(111)

        if title is not None:
            ax.set_title(title)

        ax.invert_yaxis()
        frame = pandas.DataFrame(
            list(zip(layers, values)), columns=["Layer", "Sensitivity"]
        )
        frame.plot.barh(ax=ax, x="Layer", y="Sensitivity")
        plt.gca().invert_yaxis()

        if path is None:
            plt.show()

            return fig, ax

        path = clean_path(path)
        create_parent_dirs(path)
        plt.savefig(path)
        plt.close(fig)

        return None, None

    def __init__(self, module: Module, data: Dataset, loss_fn: LossWrapper):
        """
        :param module: the module to run the kernel sparsity sensitivity analysis over
                       will extract all prunable layers out
        :param data: the data to run through the module for calculating the sensitivity analysis
        :param loss_fn: the loss function to use for the sensitivity analysis
        """
        self._module = module
        self._data = data
        self._loss_fn = loss_fn
        self._progress_hooks = OrderedDict()

    def register_progress_hook(
        self, hook: Callable[[KSSensitivityProgress], None]
    ) -> RemovableHandle:
        """
        :param hook: the hook to be called after each progress event
        :return: a handle to remove the reference after done subscribing
        """
        handle = RemovableHandle(self._progress_hooks)
        self._progress_hooks[
            handle.id
        ] = hook  # type: Dict[Any, Callable[[KSSensitivityProgress], None]]

        return handle

    def run_one_shot(
        self,
        device: str,
        batch_size: int,
        samples_per_measurement: int,
        sparsity_levels: List[int] = None,
        cache_data: bool = True,
        loader_args: Dict = None,
        data_loader_const: Callable = DataLoader,
        tester_run_funcs: ModuleRunFuncs = None,
    ):
        """
        Run a one shot sensitivity analysis for kernel sparsity
        It does not retrain, and instead puts the model to eval mode.
        Moves layer by layer to calculate the sensitivity analysis for each and resets the previously run layers

        :param device: the device to run the analysis on; ex: cpu, cuda, cuda:0,1
        :param batch_size: the batch size to run through the model in eval mode
        :param samples_per_measurement: the number of samples or items to take for each measurement at each sparsity lev
        :param sparsity_levels: the sparsity levels to check for each layer to calculate sensitivity
        :param cache_data: True to cache the data in CPU RAM instead of reloading from disk
                           False to not cache. Note, num_workers must be 0 for this to work, so first load is slow
        :param loader_args: the arguments to supply to the DataLoader other than the dataset and batch size
        :param data_loader_const: the constructor used to create a DataLoader instance, defaults to the regular pytorch
        :param tester_run_funcs: override functions to use in the ModuleTester that runs
        :return: the sensitivity results for every layer that is prunable
        """
        data = (
            EarlyStopDataset(self._data, samples_per_measurement)
            if len(self._data) > samples_per_measurement > 0
            else self._data
        )

        if loader_args is None:
            loader_args = {}

        if cache_data:
            # cacheable dataset does not work with parallel data loaders
            if "num_workers" in loader_args and loader_args["num_workers"] != 0:
                raise ValueError("num_workers must be 0 for dataset cache")

            loader_args["num_workers"] = 0
            data = CacheableDataset(data)

        if sparsity_levels is None:
            sparsity_levels = [0.05, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]

        module, device, device_ids = model_to_device(self._module, device)
        analyzer = ModuleAnalyzer(module, enabled=True)
        device_str = (
            device
            if device_ids is None or len(device_ids) < 2
            else "{}:{}".format(device, device_ids[0])
        )

        layers = {}
        layers.update(get_conv_layers(module))
        layers.update(get_linear_layers(module))
        progress = KSSensitivityProgress(
            layer_index=-1,
            layer_name="",
            layers=list(layers.keys()),
            sparsity_index=-1,
            sparsity_levels=sparsity_levels,
            measurement_step=-1,
            samples_per_measurement=samples_per_measurement,
        )

        def _batch_end(
            _epoch: int,
            _step: int,
            _batch_size: int,
            _data: Any,
            _pred: Any,
            _losses: Any,
        ):
            progress.measurement_step = _step
            analyzer.enabled = False
            self._invoke_progress_hooks(progress)

        tester = ModuleTester(module, device_str, self._loss_fn)
        batch_end_hook = tester.run_hooks.register_batch_end_hook(_batch_end)
        if tester_run_funcs is not None:
            tester.run_funcs.copy(tester_run_funcs)

        sensitivities = []

        for layer_index, (name, layer) in enumerate(layers.items()):
            progress.layer_index = layer_index
            progress.layer_name = name
            progress.sparsity_index = -1
            progress.measurement_step = -1
            self._invoke_progress_hooks(progress)

            sparsities_loss = []
            mask = ModuleParamKSMask(layer, store_init=True)
            mask.enabled = True

            for sparsity_index, sparsity_level in enumerate(sparsity_levels):
                progress.sparsity_index = sparsity_index
                progress.measurement_step = -1
                self._invoke_progress_hooks(progress)

                mask.set_param_mask_from_sparsity(sparsity_level)
                data_loader = data_loader_const(data, batch_size, **loader_args)
                res = tester.run(
                    data_loader, desc="", show_progress=False, track_results=True
                )
                sparsities_loss.append(
                    (sparsity_level, res.result_mean(DEFAULT_LOSS_KEY).item())
                )

            mask.enabled = False
            mask.reset()
            del mask

            desc = analyzer.layer_desc(name)
            sensitivities.append(
                ModuleParamKSSensitivity(
                    name, "weight", desc.type_, desc.execution_order, sparsities_loss
                )
            )

        batch_end_hook.remove()
        sensitivities.sort(key=lambda val: val.execution_order)

        return sensitivities

    def _invoke_progress_hooks(self, progress: KSSensitivityProgress):
        for hook in self._progress_hooks.values():
            hook(progress)
