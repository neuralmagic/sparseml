from typing import Callable, List, Tuple, Any
import math
import ipywidgets as widgets
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset

from neuralmagicML.recal import one_shot_sensitivity_analysis

__all__ = []


class ConfigCreatorSettings(object):
    def __init__(self):
        self.model = None
        self.model_name = None
        self.dataset = None
        self.sample_input = None
        self.loss_fn = None


class ModelSensitivityAnalysisWidget(object):
    CAT_UNIFORM = 'uniform'
    CAT_EQN_APPROX = 'eqn approx'
    CAT_STATIC_APPROX = 'static approx'

    EQN_APPROX_IS_PARAMS = 'is_params'
    EQN_APPROX_IS_KERNELS = 'is_kernels'
    EQN_APPROX_ER = 'er'
    EQN_APPROX_ERK = 'erk'

    def __init__(self, sens_updater: Callable[[List[Tuple[str, float]]], None], model: Module,
                 sample_input: Tuple[Tensor, ...], dataset: Dataset, loss_fn: Callable):
        self._sens_updater = sens_updater
        self._model = model
        self._sample_input = sample_input
        self._dataset = dataset
        self._loss_fn = loss_fn

        self._fig_axes = None
        self._approx_sensitivities = None
        self._static_sensitivities = None

        self._sens_cat_buttons = None
        self._desc_text = None
        self._error_container = None
        self._error_text = None
        self._cat_body = None
        self._cat_bodies = None
        self._fig_output = None

        self._eqn_approx_buttons = None
        self._eqn_approx_desc_text = None

        self._static_approx_buttons_container = None
        self._static_approx_device_buttons = None
        self._static_approx_batch_size_slider = None
        self._static_approx_num_samples_slider = None
        self._static_approx_cache_check = None
        self._static_approx_gen_button = None

        self._cat_type = None
        self._eqn_approx_type = None

    def create(self) -> widgets.Box:
        self._sens_cat_buttons = widgets.ToggleButtons(
            options=[ModelSensitivityAnalysisWidget.CAT_UNIFORM,
                     ModelSensitivityAnalysisWidget.CAT_EQN_APPROX,
                     ModelSensitivityAnalysisWidget.CAT_STATIC_APPROX],
            description='Sensitivity:',
            disabled=False,
            button_style='info'
        )
        self._desc_text = widgets.HTML(value='', description='')
        self._error_text = widgets.HTML(value='', description='Error')
        self._error_container = widgets.Box(())

        self._eqn_approx_buttons = widgets.ToggleButtons(
            options=[ModelSensitivityAnalysisWidget.EQN_APPROX_IS_PARAMS,
                     ModelSensitivityAnalysisWidget.EQN_APPROX_IS_KERNELS,
                     ModelSensitivityAnalysisWidget.EQN_APPROX_ERK,
                     ModelSensitivityAnalysisWidget.EQN_APPROX_ER],
            description='Equation:',
            disabled=False,
            button_style=''
        )
        self._eqn_approx_desc_text = widgets.HTML(value='', description='')

        devices = [*['cuda:{}'.format(dev_id)
                     for dev_id in range(torch.cuda.device_count()) if torch.cuda.is_available()],
                   'cpu']
        self._static_approx_device_buttons = {
            device: widgets.ToggleButton(
                value=False,
                description=device,
                disabled=False,
                button_style=''
            ) for device in devices
        }
        self._static_approx_device_buttons[devices[0]].value=True
        self._static_approx_buttons_container = widgets.HBox((
            widgets.Label(value='Device'),
            widgets.VBox(tuple([val for (_, val) in self._static_approx_device_buttons.items()]))
        ))
        self._static_approx_batch_size_slider = widgets.FloatLogSlider(
            value=64,
            base=2,
            min=0,
            max=12,
            step=1,
            description='Batch Size:'
        )
        self._static_approx_num_samples_slider = widgets.FloatLogSlider(
            value=min(64, len(self._dataset) if self._dataset is not None else 1),
            base=2,
            min=0,
            max=math.log2(len(self._dataset)) if self._dataset is not None else 0,
            step=1,
            description='Num Samples:'
        )
        self._static_approx_cache_check = widgets.Checkbox(
            value=True,
            description='Cache data in CPU RAM',
            disabled=False
        )
        self._static_approx_gen_button = widgets.Button(
            description='Run Approx',
            disabled=False,
            button_style='info'
        )

        self._cat_body = widgets.Box(())
        self._cat_bodies = {
            ModelSensitivityAnalysisWidget.CAT_UNIFORM: widgets.VBox(()),
            ModelSensitivityAnalysisWidget.CAT_EQN_APPROX: widgets.VBox((
                self._eqn_approx_buttons, self._eqn_approx_desc_text
            )),
            ModelSensitivityAnalysisWidget.CAT_STATIC_APPROX: widgets.VBox((
                self._static_approx_buttons_container, self._static_approx_cache_check,
                self._static_approx_batch_size_slider, self._static_approx_num_samples_slider,
                self._static_approx_gen_button
            )),
        }
        self._fig_output = widgets.Output(layout={'border': '1px solid black'})

        self._create_cat_buttons_observer()
        self._create_eqn_approx_buttons_observer()
        self._create_static_approx_buttons_observer()
        self._create_static_approx_run_observer()

        return widgets.VBox((
            self._sens_cat_buttons,
            self._desc_text,
            self._error_container,
            self._cat_body,
            self._fig_output
        ))

    def _create_cat_buttons_observer(self):
        def _updater(cat_type):
            self._cat_type = cat_type
            self._update_description()
            self._update_cat_body()
            self._update_figure()

            if self._cat_type == ModelSensitivityAnalysisWidget.CAT_EQN_APPROX:
                self._run_approx_sensitivities()

        def _trigger(change):
            _updater(change['new'])

        self._sens_cat_buttons.observe(_trigger, names='value')
        _updater(self._sens_cat_buttons.value)

    def _create_eqn_approx_buttons_observer(self):
        def _updater(eqn_approx_type):
            self._eqn_approx_type = eqn_approx_type
            self._update_eqn_approx_description()
            self._run_approx_sensitivities()

        def _trigger(change):
            _updater(change['new'])

        self._eqn_approx_buttons.observe(_trigger, names='value')
        _updater(self._eqn_approx_buttons.value)

    def _create_static_approx_buttons_observer(self):
        def _updater(selected: bool, device: str):
            if selected and device == 'cpu':
                for _key, _val in self._static_approx_device_buttons.items():
                    if _key != 'cpu':
                        _val.value = False
            elif selected:
                for _key, _val in self._static_approx_device_buttons.items():
                    if _key == 'cpu':
                        _val.value = False

        def _trigger(change):
            _updater(change['new'], change['owner'].description)

        for key, val in self._static_approx_device_buttons.items():
            val.observe(_trigger, names='value')

    def _create_static_approx_run_observer(self):
        def _click(change):
            self._static_sensitivities = None
            self._clear_figure()
            self._run_static_sensitivities()

        self._static_approx_gen_button.on_click(_click)

    def _update_error(self, error: Any):
        if not error:
            self._error_container.children = ()
            self._error_text.value = ''

            return

        self._error_container.children = (self._error_text,)
        self._error_text.value = '<span style="color: red">{}<span>'.format(error)

    def _update_description(self):
        if self._cat_type == ModelSensitivityAnalysisWidget.CAT_UNIFORM:
            self._desc_text.value = 'Uniform distribution, all layers are considered to have the same sensitivity. ' \
                                    'So, applies the same sparsity percentage to every layer'

        if self._cat_type == ModelSensitivityAnalysisWidget.CAT_EQN_APPROX:
            self._desc_text.value = 'Approximate the sensitivity of each layer with equations based on the architecture'

        if self._cat_type == ModelSensitivityAnalysisWidget.CAT_STATIC_APPROX:
            self._desc_text.value = 'Approximate by running a sensitivity analysis independently for each layer ' \
                                    'statically (ie without retraining)'

    def _update_cat_body(self):
        self._cat_body.children = ()

        if self._cat_type in self._cat_bodies:
            self._cat_body.children = (self._cat_bodies[self._cat_type],)

    def _update_eqn_approx_description(self):
        if self._eqn_approx_type == ModelSensitivityAnalysisWidget.EQN_APPROX_IS_PARAMS:
            self._eqn_approx_desc_text.value = 'Sensitivity based on the number of parameters in each layer ' \
                                               'and the amount of change in capacity for the input and output space. ' \
                                               'Larger param spaces that limit change in capacity ' \
                                               'are considered less sensitive'

        if self._eqn_approx_type == ModelSensitivityAnalysisWidget.EQN_APPROX_IS_KERNELS:
            self._eqn_approx_desc_text.value = 'Sensitivity based on the convolutional kernel size in each layer ' \
                                               'and the amount of change in capacity for the input and output space. ' \
                                               'Larger kernel spaces that limit change in capacity ' \
                                               'are considered less sensitive'

        if self._eqn_approx_type == ModelSensitivityAnalysisWidget.EQN_APPROX_ERK:
            self._eqn_approx_desc_text.value = 'Sensitivity based on the number of input and output channels ' \
                                               'and kernel size in each layer as found in ' \
                                               '<a href="https://arxiv.org/pdf/1911.11134.pdf" ' \
                                               '_target="blank">this paper</a>, ' \
                                               'larger channel and kernel spaces are considered less sensitive'

        if self._eqn_approx_type == ModelSensitivityAnalysisWidget.EQN_APPROX_ER:
            self._eqn_approx_desc_text.value = 'Sensitivity based on the number of input and output channels ' \
                                               'in each layer as found in ' \
                                               '<a href="https://www.nature.com/articles/s41467-018-04316-3" ' \
                                               '_target="blank">this paper</a>,' \
                                               ' larger channel spaces are considered less sensitive'

    def _run_approx_sensitivities(self):
        if (self._model is not None and self._approx_sensitivities is None and
                self._cat_type != ModelSensitivityAnalysisWidget.CAT_STATIC_APPROX):
            self._approx_sensitivities = approx_sensitivity_analysis(self._model, self._sample_input)

        self._update_figure()

    def _run_static_sensitivities(self):
        if (self._model is not None and self._dataset is not None and self._static_sensitivities is None and
                self._cat_type == ModelSensitivityAnalysisWidget.CAT_STATIC_APPROX):
            devices = [key for key, val in self._static_approx_device_buttons.items() if val.value]
            device = devices[0] if len(devices) < 2 else 'cuda:{}'.format(','.join(device[-1] for device in devices))
            batch_size = int(self._static_approx_batch_size_slider.value)
            samples_per_check = int(self._static_approx_num_samples_slider.value)
            cache = self._static_approx_cache_check.value
            self._static_sensitivities = one_shot_sensitivity_analysis(self._model, self._dataset, self._loss_fn,
                                                                       device, batch_size, samples_per_check,
                                                                       cache_data=cache)

        self._update_figure()

    def _update_figure(self):
        self._clear_figure()

        sens = None
        title = None

        if self._cat_type == ModelSensitivityAnalysisWidget.CAT_UNIFORM and self._approx_sensitivities:
            sens = [(sens.layer_desc.name, sens.uniform) for sens in self._approx_sensitivities]

        if self._cat_type == ModelSensitivityAnalysisWidget.CAT_EQN_APPROX and self._approx_sensitivities:
            if self._eqn_approx_type == ModelSensitivityAnalysisWidget.EQN_APPROX_IS_PARAMS:
                sens = [(sens.layer_desc.name, sens.is_params) for sens in self._approx_sensitivities]

            if self._eqn_approx_type == ModelSensitivityAnalysisWidget.EQN_APPROX_IS_KERNELS:
                sens = [(sens.layer_desc.name, sens.is_kernels) for sens in self._approx_sensitivities]

            if self._eqn_approx_type == ModelSensitivityAnalysisWidget.EQN_APPROX_ERK:
                sens = [(sens.layer_desc.name, sens.erk) for sens in self._approx_sensitivities]

            if self._eqn_approx_type == ModelSensitivityAnalysisWidget.EQN_APPROX_ER:
                sens = [(sens.layer_desc.name, sens.er) for sens in self._approx_sensitivities]

        if self._cat_type == ModelSensitivityAnalysisWidget.CAT_STATIC_APPROX and self._static_sensitivities:
            sens = [(sens.layer_desc.name, sens.integrate) for sens in self._static_sensitivities]

        if sens is not None:
            self._fig_axes = plot_sensitivities(sens, title)

            with self._fig_output:
                plt.show()

        self._fire_new_sens(sens)

    def _clear_figure(self):
        if self._fig_axes is not None:
            plt.close(self._fig_axes[0])
            self._fig_axes = None
            self._fig_output.clear_output()

    def _fire_new_sens(self, sens: List[Tuple[str, float]]):
        self._sens_updater(sens)


