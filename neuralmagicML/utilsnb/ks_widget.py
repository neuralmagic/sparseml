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
from neuralmagicML.utilsnb.helpers import format_html


__all__ = [
    "KSWidgetContainer",
    "PruningEpochWidget",
    "PruneLayerWidget",
    "PruningLayersWidget",
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


class PruneLayerWidget(_Widget):
    """
    Widget used in KS notebooks for setting up pruning hyperparams
    for an individual layer. Includes visual data for identifying the layer
    such as name and description. Also allows pruning for the layer to
    be enabled or disabled. Finally, allows the user to set the desired
    final sparsity for the layer.

    :param name: name of the layer that can be selected for pruning
    :param desc: optional description of the layer to display for more context
    :param enabled: optional pre configured settings for enabling pruning on widget
        display, default True
    :param end_sparsity: optional pre configured setting for the sparsity to set
        for each layer
        on display, by default all layers are set to 0.8
    :param loss_sens_analysis: optional sensitivity analysis to use to display next
        to the layers
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
            summary = self._loss_sens_analysis.results_summary(normalize=True)
            layer_index = summary["layers"].index(self._name)
            layer_integral = summary["values"][layer_index]
            bounds = (summary["max"] - summary["min"]) * 0.25

            sens_visualization = widgets.FloatProgress(
                value=layer_integral - summary["min"] + bounds,
                min=0.0,
                max=summary["max"] - summary["min"] + 2 * bounds,
                bar_style="info" if layer_integral <= 0.0 else "warning",
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
                    layers=[self._name],
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
                    layers=[self._name],
                    init_sparsity=0.05,
                    final_sparsity=self._end_sparsity,
                    start_epoch=0.0,
                    end_epoch=1.0,
                    update_frequency=1.0,
                )
            ]

        raise ValueError("unknown framework given of {}".format(framework))


class PruningLayersWidget(_Widget):
    """
    Widget used in KS notebooks for setting up pruning hyperparams
    for multiple layers in a model.
    See :py:func `~PruneLayerWidget` for more details

    :param layer_names: names of the layers that can be selected for pruning
    :param layer_descs: optional descriptions of the layers to display for more context
    :param layer_enables: optional pre configured settings for which layers to
        enable on display, by default all layers are enabled
    :param layer_sparsities: optional pre configured settings for the sparsity to
        set for each layer on display, by default all layers are set to 0.8
    :param loss_sens_analysis: optional sensitivity analysis to use to display next
        to the layers
    """

    def __init__(
        self,
        layer_names: List[str],
        layer_descs: Union[List[str], None] = None,
        layer_enables: Union[List[bool], None] = None,
        layer_sparsities: Union[List[float], None] = None,
        loss_sens_analysis: Union[KSLossSensitivityAnalysis, None] = None,
    ):
        if not layer_names or len(layer_names) < 1:
            raise ValueError("layer_names must be provided")

        if layer_descs is not None and len(layer_descs) != len(layer_names):
            raise ValueError(
                (
                    "layer_descs given of length {}, "
                    "must match layer_names length {}"
                ).format(len(layer_descs), len(layer_names))
            )

        if layer_enables is not None and len(layer_enables) != len(layer_names):
            raise ValueError(
                (
                    "layer_enables given of length {}, "
                    "must match layer_names length {}"
                ).format(len(layer_enables), len(layer_names))
            )

        if layer_sparsities is not None and len(layer_sparsities) != len(layer_names):
            raise ValueError(
                (
                    "layer_sparsities given of length {}, "
                    "must match layer_names length {}"
                ).format(len(layer_sparsities), len(layer_names))
            )

        if loss_sens_analysis:
            for name in layer_names:
                try:
                    assert loss_sens_analysis.get_result(name)
                except Exception as ex:
                    raise ValueError(
                        "layer name {} must be in the loss_sens_analysis: {}".format(
                            name, ex
                        )
                    )

        self._layer_widgets = [
            PruneLayerWidget(
                name,
                enabled=layer_enables[index] if layer_enables else True,
                desc=layer_descs[index] if layer_descs else None,
                end_sparsity=layer_sparsities[index] if layer_sparsities else 0.8,
                loss_sens_analysis=loss_sens_analysis,
            )
            for index, name in enumerate(layer_names)
        ]
        self._layer_descs = layer_descs
        self._layer_names = layer_names
        self._loss_sens_analysis = loss_sens_analysis

    def create(self):
        """
        :return: a created ipywidget for display
        """
        return widgets.VBox(
            (
                widgets.HTML(value=format_html("Layer Sparsity Setter", header="h3")),
                *[widg.create() for widg in self._layer_widgets],
            )
        )

    def get_modifiers(self, framework: str) -> List:
        """
        :param framework: the ML framework to get the modifiers for
        :return: the list of modifiers for the given config settings
        """
        modifiers = []

        for widg in self._layer_widgets:
            modifiers.extend(widg.get_modifiers(framework))

        return modifiers


class KSWidgetContainer(object):
    """
    Widget used in KS notebooks for setting up pruning hyperparams.
    See :py:func `~PruneLayersWidget` and :py:func `~PruningEpochWidget`
    for more details

    :param epoch_widget: the instance of the epoch widget to display
    :param layers_widget: the instance of the layers widget to display
    """

    def __init__(
        self, epoch_widget: PruningEpochWidget, layers_widget: PruningLayersWidget
    ):
        if not epoch_widget:
            raise ValueError("epoch_widget must be supplied")

        if not layers_widget:
            raise ValueError("layers_widget must be supplied")

        self._epoch_widget = epoch_widget
        self._layers_widget = layers_widget

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
                self._layers_widget.create(),
            )
        )

    def get_manager(self, framework: str):
        """
        :param framework: the ML framework to create the modifiers and manager for
        :return: the created manager containing the modifiers with the desired settings
        """
        epoch_modifiers = self._epoch_widget.get_modifiers(framework)
        layers_modifiers = self._layers_widget.get_modifiers(framework)

        for mod in layers_modifiers:
            mod.end_epoch = self._epoch_widget.end_epoch
            mod.start_epoch = self._epoch_widget.start_epoch
            mod.update_frequency = self._epoch_widget.update_frequency

        modifiers = []
        modifiers.extend(epoch_modifiers)
        modifiers.extend(layers_modifiers)

        if framework == PYTORCH_FRAMEWORK:
            return ScheduledModifierManager_pt(modifiers)

        if framework == TENSORFLOW_FRAMEWORK:
            return ScheduledModifierManager_tf(modifiers)

        raise ValueError("unknown framework given of {}".format(framework))
