from typing import List, Tuple, Callable, Dict, Any, Union
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import pandas

from torch.nn import Module
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torch.utils.hooks import RemovableHandle

from ...utils import (
    LossWrapper,
    DEFAULT_LOSS_KEY,
    ModuleTrainer,
    ModuleRunFuncs,
    ModuleRunResults,
    clean_path,
    create_parent_dirs,
)
from ...models import model_to_device


__all__ = ["LRSensitivityProgress", "ModuleLRSensitivityAnalysis"]


def _measured_data_loader(data_loader: DataLoader, num_yields: int):
    counter = 0
    while True:
        for data in data_loader:
            yield data

            counter += 1
            if counter == num_yields:
                return


class LRSensitivityProgress(object):
    """
    Simple class for tracking the progress of a sensitivity analysis
    """

    def __init__(
        self,
        lr_index: int,
        lr: float,
        check_lrs: List[float],
        batch: int,
        batches_per_measurement: int,
    ):
        """
        :param lr_index: the current index of the learning rate being analyzed
        :param lr: the current learning rate being analyzed
        :param check_lrs: the list of learning rates to be analyzed in order
        :param batch: the current batch for the given lr that is being analyzed
        :param batches_per_measurement: the number of batches to measure per each learning rate
        """
        self.lr_index = lr_index
        self.lr = lr
        self.check_lrs = check_lrs
        self.batch = batch
        self.batches_per_measurement = batches_per_measurement

    def __repr__(self):
        return "{}(lr_index={}, lr={}, check_lrs={}, batch={}, batches_per_measurement={})".format(
            self.__class__.__name__,
            self.lr_index,
            self.lr,
            self.check_lrs,
            self.batch,
            self.batches_per_measurement,
        )


class ModuleLRSensitivityAnalysis(object):
    """
    Class for handling running sensitivity analysis for learning rates on modules
    """

    @staticmethod
    def save_sensitivities_json(
        sensitivities: List[Tuple[float, ModuleRunResults]], path: str
    ):
        """
        :param sensitivities: the measured sensitivities
        :param path: the path to save the json file at representing the lr sensitivities
        """
        path = clean_path(path)
        create_parent_dirs(path)
        sens_object = {
            "lr_sensitivities": [
                (
                    sens[0],
                    {
                        loss: sens[1].result_mean(loss).item()
                        for loss in sens[1].results.keys()
                    },
                )
                for sens in sensitivities
            ]
        }

        with open(path, "w") as file:
            json.dump(sens_object, file)

    @staticmethod
    def plot_sensitivities(
        sensitivities: List[Tuple[float, ModuleRunResults]],
        path: Union[str, None],
        plot_loss_key: str = DEFAULT_LOSS_KEY,
        title: Union[str, None] = "__default__",
    ) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[None, None]]:
        """
        :param sensitivities: the learning rate sensitivities to plot
        :param path: the path for where to save the plot, if not supplied will display it
        :param plot_loss_key: the loss key to use for plotting, defaults to DEFAULT_LOSS_KEY
        :param title: the title of the plot to apply, defaults to '{plot_loss_key} LR Sensitivity'
        :return: the figure and axes if the figure was displayed; else None, None
        """
        analysis = [
            (sens[0], sens[1].result_mean(plot_loss_key)) for sens in sensitivities
        ]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        if title is None:
            title = ""
        elif title == "__default__":
            title = "{} LR Sensitivity".format(DEFAULT_LOSS_KEY)

        ax.set_title(title)
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Avg Loss")
        frame = pandas.DataFrame.from_records(
            analysis, columns=["Learning Rate", "Avg Loss"]
        )
        frame.plot(x="Learning Rate", y="Avg Loss", marker=".", logx=True, ax=ax)

        if path is None:
            plt.show()

            return fig, ax

        path = clean_path(path)
        create_parent_dirs(path)
        plt.savefig(path)
        plt.close(fig)

        return None, None

    @staticmethod
    def default_exponential_check_lrs(
        init_lr: float = 1e-9, final_lr: float = 1e0, lr_mult: float = 1.1
    ):
        """
        :param init_lr: the initial learning rate in the returned list
        :param final_lr: the final learning rate in the returned list
        :param lr_mult: the multiplier increase for each step between init_lr and final_lr
        :return: the list of created lrs that increase exponentially between init_lr and final_lr according to lr_mult
        """
        check_lrs = [init_lr]

        while check_lrs[-1] < final_lr:
            check_lrs.append(check_lrs[-1] * lr_mult)

        check_lrs.append(final_lr)

        return check_lrs

    def __init__(self, module: Module, data: Dataset, loss_fn: LossWrapper):
        """
        :param module: the module to run the learning rate sensitivity analysis over
        :param data: the data to run through the module for calculating the sensitivity analysis
        :param loss_fn: the loss function to use for the sensitivity analysis
        """
        self._module = module
        self._data = data
        self._loss_fn = loss_fn
        self._progress_hooks = OrderedDict()

    def register_progress_hook(
        self, hook: Callable[[LRSensitivityProgress], None]
    ) -> RemovableHandle:
        """
        :param hook: the hook to be called after each progress event
        :return: a handle to remove the reference after done subscribing
        """
        handle = RemovableHandle(self._progress_hooks)
        self._progress_hooks[handle.id] = hook

        return handle

    def run(
        self,
        device: str,
        batch_size: int,
        batches_per_measurement: int,
        check_lrs: List[float],
        sgd_args: Dict = None,
        loader_args: Dict = None,
        data_loader_const: Callable = DataLoader,
        trainer_run_funcs: ModuleRunFuncs = None,
    ) -> List[Tuple[float, ModuleRunResults]]:
        """
        :param device: the device to run the analysis on; ex: cpu, cuda, cuda:0,1
        :param batch_size: the batch size to run through them model for the analysis
        :param batches_per_measurement: the number of batches to run through for the analysis at each LR
        :param check_lrs: the learning rates to check for analysis (will sort them small to large before running)
        :param sgd_args: any args to add to the SGD optimizer that will be created for analysis
        :param loader_args: any args to add to the DataLoader
        :param data_loader_const: a data loader constructor to create the data loader with, default is DataLoader
        :param trainer_run_funcs: override functions for ModuleTrainer class
        :return: a list of tuples containing the analyzed learning rate at 0 and the ModuleRunResults in 1,
                 ModuleRunResults being a collection of all the batch results run through the module at that LR
        """
        if loader_args is None:
            loader_args = {}

        if sgd_args is None:
            sgd_args = {}

        module, device, device_ids = model_to_device(self._module, device)
        device_str = (
            device
            if device_ids is None or len(device_ids) < 2
            else "{}:{}".format(device, device_ids[0])
        )

        check_lrs = sorted(check_lrs)
        optim = SGD(module.parameters(), lr=1.0, **sgd_args)
        results = []  # type: List[Tuple[float, ModuleRunResults]]

        progress = LRSensitivityProgress(
            lr_index=-1,
            lr=-1,
            check_lrs=check_lrs,
            batch=-1,
            batches_per_measurement=batches_per_measurement,
        )
        self._invoke_progress_hooks(progress)

        def _batch_end(
            _epoch: int,
            _step: int,
            _batch_size: int,
            _data: Any,
            _pred: Any,
            _losses: Any,
        ):
            if progress.lr_index != -1:
                progress.batch += 1
                results[-1][1].append(_losses, _batch_size)

            if (
                progress.batch == -1
                or progress.batch + 1 == progress.batches_per_measurement
            ):
                progress.batch = 0
                progress.lr_index += 1
                progress.lr = progress.check_lrs[progress.lr_index]
                results.append((progress.lr, ModuleRunResults()))

                for param_group in self.param_groups:
                    param_group["lr"] = progress.lr

            self._invoke_progress_hooks(progress)

        trainer = ModuleTrainer(module, device_str, self._loss_fn, optim)
        batch_end_hook = trainer.run_hooks.register_batch_end_hook(_batch_end)
        if trainer_run_funcs is not None:
            trainer.run_funcs.copy(trainer_run_funcs)

        data_loader = data_loader_const(
            self._data, batch_size=batch_size, **loader_args
        )
        data_loader = _measured_data_loader(
            data_loader, len(check_lrs) * batches_per_measurement
        )
        trainer.run(data_loader, desc="")
        batch_end_hook.remove()

        return results

    def _invoke_progress_hooks(self, progress: LRSensitivityProgress):
        for hook in self._progress_hooks.values():
            hook(progress)
