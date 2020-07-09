"""
Code related to kernel sparsity widget display in a jupyter notebook using ipywidgets
"""

from abc import ABC, abstractmethod
from typing import List, Union
import ipywidgets as widgets

from neuralmagicML.recal import KSLossSensitivityAnalysis
from neuralmagicML.pytorch.recal import (
    PYTORCH_FRAMEWORK,
    EpochRangeModifier as EpochRangeModifier_pt,
    GradualKSModifier as GradualKSModifier_pt,
    ScheduledModifierManager as ScheduledModifierManager_pt,
)
from neuralmagicML.tensorflow.recal import (
    TENSORFLOW_FRAMEWORK,
    EpochRangeModifier as EpochRangeModifier_tf,
    GradualKSModifier as GradualKSModifier_tf,
    ScheduledModifierManager as ScheduledModifierManager_tf,
)
from neuralmagicML.tensorflow.utils import clean_tensor_name
from neuralmagicML.utilsnb.helpers import format_html


__all__ = [
    "KSWidgetContainer",
    "PruningEpochWidget",
    "PruneParamWidget",
    "PruningParamsWidget",
]


class _Widget(ABC):
    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def get_modifiers(self, framework: str) -> List:
        pass


class PruningEpochWidget(_Widget):
    """
    Widget used in KS notebooks for setting up pruning epoch hyperparams such as
    start and end epoch for pruning, the frequency to prune,
    and the total epochs to train for

    :param start_epoch: the default epoch to start pruning at
    :param end_epoch: the default epoch to end pruning at
    :param total_epochs: the default number of epochs to train for
    :param max_epochs: the maximum number of epochs that can be used
    :param update_frequency: the default update frequency for pruning
    """

    def __init__(
        self,
        start_epoch: int,
        end_epoch: int,
        total_epochs: int,
        max_epochs: int,
        update_frequency: float = 1.0,
    ):
        self._start_epoch = -1
        self._end_epoch = -1
        self._total_epochs = -1
        self._max_epochs = max_epochs
        self._update_frequency = update_frequency

        self.max_epochs = max_epochs
        self.total_epochs = total_epochs
        self.end_epoch = end_epoch
        self.start_epoch = start_epoch
        self.update_frequency = update_frequency

    @property
    def start_epoch(self) -> int:
        """
        :return: the epoch to start pruning at
        """
        return self._start_epoch

    @start_epoch.setter
    def start_epoch(self, value: int):
        """
        :param value: the epoch to start pruning at
        """
        if value >= self._end_epoch:
            raise ValueError(
                "value of {} cannot be greater than or equal to end_epoch {}".format(
                    value, self._end_epoch
                )
            )

        self._start_epoch = value

    @property
    def end_epoch(self) -> int:
        """
        :return: the epoch to end pruning at
        """
        return self._end_epoch

    @end_epoch.setter
    def end_epoch(self, value):
        """
        :param value: the epoch to end pruning at
        """
        if value > self._total_epochs:
            raise ValueError(
                "value of {} cannot be greater than total_epochs {}".format(
                    value, self._total_epochs
                )
            )

        if value <= self._start_epoch:
            raise ValueError(
                "value of {} must be greater than start_epoch {}".format(
                    value, self._start_epoch
                )
            )

        self._end_epoch = value

    @property
    def total_epochs(self) -> int:
        """
        :return: the number of epochs to train for
        """
        return self._total_epochs

    @total_epochs.setter
    def total_epochs(self, value: int):
        """
        :param value: the number of epochs to train for
        """
        if value > self._max_epochs:
            raise ValueError(
                "value of {} cannot be greater than max_epochs {}".format(
                    value, self._max_epochs
                )
            )

        if value < self._end_epoch:
            raise ValueError(
                "value of {} must be greater than or equal to end_epoch {}".format(
                    value, self._end_epoch
                )
            )

        self._total_epochs = value

    @property
    def max_epochs(self) -> int:
        """
        :return: the maximum number of epochs that can be used
        """
        return self._max_epochs

    @max_epochs.setter
    def max_epochs(self, value: int):
        """
        :param value: the maximum number of epochs that can be used
        """
        if value < self._max_epochs:
            raise ValueError(
                "value of {} must be greater than or equal to total_epochs {}".format(
                    value, self._max_epochs
                )
            )

        self._max_epochs = value

    @property
    def update_frequency(self) -> float:
        """
        :return: the update frequency for pruning
        """
        return self._update_frequency

    @update_frequency.setter
    def update_frequency(self, value: float):
        """
        :param value: the update frequency for pruning
        """
        self._update_frequency = value

    def create(self):
        """
        :return: a created ipywidget for display
        """
        total_epochs_slider = widgets.IntSlider(
            value=self.total_epochs, min=1, max=self.max_epochs
        )
        pruning_epochs_slider = widgets.IntRangeSlider(
            value=[self.start_epoch, self.end_epoch], min=0, max=self.total_epochs
        )
        update_frequency_slider = widgets.FloatSlider(
            value=self._update_frequency, min=0.0, max=1.0
        )

        def _update_pruning():
            pruning_epochs_slider.value = [self.start_epoch, self.end_epoch]

        def _on_total_change(change):
            new_total = change["new"]

            if self.end_epoch >= new_total or self.start_epoch >= new_total:
                end_epoch = self.end_epoch if new_total >= self.end_epoch else new_total
                start_epoch = (
                    self.start_epoch if end_epoch > self.start_epoch else end_epoch - 1
                )
                self.start_epoch = start_epoch
                self.end_epoch = end_epoch
                _update_pruning()

            self.total_epochs = new_total
            pruning_epochs_slider.max = self.total_epochs

        def _on_pruning_change(change):
            new_values = change["new"]
            start_epoch = new_values[0]
            end_epoch = new_values[1]

            if end_epoch < 1:
                self.start_epoch = 0
                self.end_epoch = 1
                _update_pruning()
            elif end_epoch <= start_epoch:
                self.start_epoch = end_epoch - 1
                self.end_epoch = end_epoch
                _update_pruning()
            elif start_epoch >= end_epoch and start_epoch + 1 > self.total_epochs:
                self.end_epoch = self.total_epochs
                self.start_epoch = self.total_epochs - 1
                _update_pruning()
            elif start_epoch >= end_epoch:
                self.end_epoch = self.start_epoch + 1
                self.start_epoch = self.end_epoch
                _update_pruning()
            else:
                self.end_epoch = end_epoch
                self.start_epoch = start_epoch

        def _on_update_frequency_change(change):
            self.update_frequency = change["new"]

        total_epochs_slider.observe(_on_total_change, names="value")
        pruning_epochs_slider.observe(_on_pruning_change, names="value")
        update_frequency_slider.observe(_on_update_frequency_change, names="value")

        return widgets.VBox(
            (
                widgets.HTML(value=format_html("Epochs Setter", header="h3")),
                widgets.HBox(
                    (widgets.Label("total training epochs:"), total_epochs_slider)
                ),
                widgets.HBox(
                    (
                        widgets.Label("pruning start and end epochs:"),
                        pruning_epochs_slider,
                    )
                ),
                widgets.HBox(
                    (
                        widgets.Label("pruning epoch update frequency:"),
                        update_frequency_slider,
                    )
                ),
            )
        )

    def get_modifiers(self, framework: str) -> List:
        """
        :param framework: the ML framework to get the modifiers for
        :return: the list of modifiers for the given config settings
        """
        if framework == PYTORCH_FRAMEWORK:
            return [EpochRangeModifier_pt(start_epoch=0.0, end_epoch=self.total_epochs)]

        if framework == TENSORFLOW_FRAMEWORK:
            return [EpochRangeModifier_tf(start_epoch=0.0, end_epoch=self.total_epochs)]

        raise ValueError("unknown framework given of {}".format(framework))


class PruneParamWidget(_Widget):
    """
    Widget used in KS notebooks for setting up pruning hyperparams
    for an individual parameter. Includes visual data for identifying the parameter
    such as name and description. Also allows pruning for the parameter to
    be enabled or disabled. Finally, allows the user to set the desired
    final sparsity for the parameter.

    :param name: name of the parameter that can be selected for pruning
    :param desc: optional description of the parameter to display for more context
    :param enabled: optional pre configured settings for enabling pruning on widget
        display, default True
    :param end_sparsity: optional pre configured setting for the sparsity to set
        for each parameter
        on display, by default all parameter are set to 0.8
    :param loss_sens_analysis: optional sensitivity analysis to use to display next
        to the parameters
    """

    def __init__(
        self,
        name: str,
        desc: str,
        enabled: bool,
        end_sparsity: float,
        loss_sens_analysis: Union[KSLossSensitivityAnalysis, None] = None,
    ):
        self._name = name
        self._desc = desc
        self._loss_sens_analysis = loss_sens_analysis
        self._enabled = enabled
        self._end_sparsity = end_sparsity

    def create(self):
        """
        :return: a created ipywidget for display
        """
        enabled_checkbox = widgets.Checkbox(value=self._enabled, indent=False)
        enabled_checkbox.layout.width = "24px"

        if self._end_sparsity < 0.25:
            self._end_sparsity = 0.25

        if self._end_sparsity > 1.0:
            self._end_sparsity = 1.0

        end_sparsity_slider = widgets.FloatSlider(
            value=self._end_sparsity,
            min=0.25,
            max=1.0,
            step=0.01,
            disabled=not self._enabled,
        )

        def _on_enabled_change(change):
            self._enabled = change["new"]
            end_sparsity_slider.disabled = not self._enabled

        def _on_end_sparsity_changed(change):
            self._end_sparsity = change["new"]

        enabled_checkbox.observe(_on_enabled_change, names="value")
        end_sparsity_slider.observe(_on_end_sparsity_changed, names="value")
        spacer = widgets.Label()
        spacer.layout.width = "24px"

        if self._loss_sens_analysis:

            param_integral = self._loss_sens_analysis.get_result(self._name)[
                "sparse_loss_integral"
            ]
            all_loss_integrals = [
                param["sparse_loss_integral"]
                for param in self._loss_sens_analysis.results
            ]
            min_val, max_val = min(all_loss_integrals), max(all_loss_integrals)
            bounds = (max_val - min_val) * 0.25

            sens_visualization = widgets.FloatProgress(
                value=param_integral - min_val + bounds,
                min=0.0,
                max=max_val - min_val + 2 * bounds,
                bar_style="info" if param_integral <= 0.0 else "warning",
                disabled=True,
            )
            sens_box = widgets.HBox(
                (widgets.HTML(format_html("loss sensitivity:")), sens_visualization,)
            )
        else:
            sens_box = widgets.Box()

        return widgets.VBox(
            (
                widgets.HBox(
                    (
                        enabled_checkbox,
                        widgets.HTML(
                            format_html("prune {}".format(self._name), header="h5")
                        ),
                    )
                ),
                widgets.HBox(
                    (
                        spacer,
                        widgets.HTML(
                            format_html(self._desc, color="gray", italic=True)
                        ),
                    )
                ),
                widgets.HBox(
                    (
                        spacer,
                        widgets.HBox(
                            (widgets.Label("desired sparsity:"), end_sparsity_slider,)
                        ),
                        sens_box,
                    )
                ),
            )
        )

    def get_modifiers(self, framework: str) -> List:
        """
        :param framework: the ML framework to get the modifiers for
        :return: the list of modifiers for the given config settings
        """
        if not self._enabled:
            return []

        if framework == PYTORCH_FRAMEWORK:
            return [
                GradualKSModifier_pt(
                    params=[self._name],
                    init_sparsity=0.05,
                    final_sparsity=self._end_sparsity,
                    start_epoch=0.0,
                    end_epoch=1.0,
                    update_frequency=1.0,
                )
            ]

        if framework == TENSORFLOW_FRAMEWORK:
            return [
                GradualKSModifier_tf(
                    params=[clean_tensor_name(self._name)],
                    init_sparsity=0.05,
                    final_sparsity=self._end_sparsity,
                    start_epoch=0.0,
                    end_epoch=1.0,
                    update_frequency=1.0,
                )
            ]

        raise ValueError("unknown framework given of {}".format(framework))


class PruningParamsWidget(_Widget):
    """
    Widget used in KS notebooks for setting up pruning hyperparams
    for multiple parameters in a model.
    See :py:func `~PruneParamsWidget` for more details

    :param param_names: names of the parameters that can be selected for pruning
    :param param_descs: optional descriptions of the parameters to display for more context
    :param param_enables: optional pre configured settings for which parameters to
        enable on display, by default all parameters are enabled
    :param param_sparsities: optional pre configured settings for the sparsity to
        set for each parameters on display, by default all parameters are set to 0.8
    :param loss_sens_analysis: optional sensitivity analysis to use to display next
        to the parameters
    """

    def __init__(
        self,
        param_names: List[str],
        param_descs: Union[List[str], None] = None,
        param_enables: Union[List[bool], None] = None,
        param_sparsities: Union[List[float], None] = None,
        loss_sens_analysis: Union[KSLossSensitivityAnalysis, None] = None,
    ):
        if not param_names or len(param_names) < 1:
            raise ValueError("param_names must be provided")

        if param_descs is not None and len(param_descs) != len(param_names):
            raise ValueError(
                (
                    "param_descs given of length {}, "
                    "must match param_names length {}"
                ).format(len(param_descs), len(param_names))
            )

        if param_enables is not None and len(param_enables) != len(param_names):
            raise ValueError(
                (
                    "param_enables given of length {}, "
                    "must match param_names length {}"
                ).format(len(param_enables), len(param_names))
            )

        if param_sparsities is not None and len(param_sparsities) != len(param_names):
            raise ValueError(
                (
                    "param_sparsities given of length {}, "
                    "must match param_names length {}"
                ).format(len(param_sparsities), len(param_names))
            )

        if loss_sens_analysis:
            for name in param_names:
                try:
                    assert loss_sens_analysis.get_result(name)
                except Exception as ex:
                    raise ValueError(
                        "param name {} must be in the loss_sens_analysis: {}".format(
                            name, ex
                        )
                    )

        self._param_widgets = [
            PruneParamWidget(
                name,
                enabled=param_enables[index] if param_enables else True,
                desc=param_descs[index] if param_descs else None,
                end_sparsity=param_sparsities[index] if param_sparsities else 0.8,
                loss_sens_analysis=loss_sens_analysis,
            )
            for index, name in enumerate(param_names)
        ]
        self._param_descs = param_descs
        self._param_names = param_names
        self._loss_sens_analysis = loss_sens_analysis

    def create(self):
        """
        :return: a created ipywidget for display
        """
        return widgets.VBox(
            (
                widgets.HTML(value=format_html("Param Sparsity Setter", header="h3")),
                *[widg.create() for widg in self._param_widgets],
            )
        )

    def get_modifiers(self, framework: str) -> List:
        """
        :param framework: the ML framework to get the modifiers for
        :return: the list of modifiers for the given config settings
        """
        modifiers = []

        for widg in self._param_widgets:
            modifiers.extend(widg.get_modifiers(framework))

        return modifiers


class KSWidgetContainer(object):
    """
    Widget used in KS notebooks for setting up pruning hyperparams.
    See :py:func `~PruneParamsWidget` and :py:func `~PruningEpochWidget`
    for more details

    :param epoch_widget: the instance of the epoch widget to display
    :param params_widget: the instance of the params widget to display
    """

    def __init__(
        self, epoch_widget: PruningEpochWidget, params_widget: PruningParamsWidget
    ):
        if not epoch_widget:
            raise ValueError("epoch_widget must be supplied")

        if not params_widget:
            raise ValueError("params_widget must be supplied")

        self._epoch_widget = epoch_widget
        self._params_widget = params_widget

    def create(self):
        """
        :return: a created ipywidget for display
        """
        return widgets.VBox(
            (
                widgets.HTML(
                    format_html("Pruning Hyperparameters Selector", header="h2")
                ),
                self._epoch_widget.create(),
                self._params_widget.create(),
            )
        )

    def get_manager(self, framework: str):
        """
        :param framework: the ML framework to create the modifiers and manager for
        :return: the created manager containing the modifiers with the desired settings
        """
        epoch_modifiers = self._epoch_widget.get_modifiers(framework)
        param_modifiers = self._params_widget.get_modifiers(framework)

        for mod in param_modifiers:
            mod.end_epoch = self._epoch_widget.end_epoch
            mod.start_epoch = self._epoch_widget.start_epoch
            mod.update_frequency = self._epoch_widget.update_frequency

        modifiers = []
        modifiers.extend(epoch_modifiers)
        modifiers.extend(param_modifiers)

        if framework == PYTORCH_FRAMEWORK:
            return ScheduledModifierManager_pt(modifiers)

        if framework == TENSORFLOW_FRAMEWORK:
            return ScheduledModifierManager_tf(modifiers)

        raise ValueError("unknown framework given of {}".format(framework))
