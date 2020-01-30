from typing import Callable, List, Tuple, Any
import os
import math
import psutil
import ipywidgets as widgets
from cpu_cores import CPUCoresCounter

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from neuralmagicML.recal import (
    ModuleAnalyzer, save_model_ks_desc, one_shot_sensitivity_analysis, save_one_shot_sensitivity_analysis
)
from neuralmagicML.utils import clean_path, create_parent_dirs

__all__ = ['ModelAnalysisWidgetSettings', 'ModelAnalysisWidget']


class ModelAnalysisWidgetSettings(object):
    def __init__(self):
        self.save_dir = None
        self.save_desc_name = None
        self.save_onnx_name = None
        self.save_sens_name = None
        self.save_perf_name = None

        self.model = None
        self.dataset = None
        self.sample_input = None
        self.loss_fn = None

        self.one_shot_sparsity_check_levels = None
        self.one_shot_loader_args = None
        self.one_shot_data_loader_const = DataLoader
        self.one_shot_tester_run_funcs = None


class ModelAnalysisWidget(object):
    def __init__(self, settings: ModelAnalysisWidgetSettings):
        self.settings = settings  # type: ModelAnalysisWidgetSettings

        self._title = None

        self._desc_container = None
        self._desc_message = None
        self._desc_button = None

        self._sens_container = None
        self._sens_device_buttons_container = None
        self._sens_device_buttons = None
        self._sens_batch_size_slider = None
        self._sens_samples_size_slider = None
        self._sens_cache_data_check = None
        self._sens_message = None
        self._sens_button = None

        self._perf_container = None
        self._perf_batch_size_slider = None
        self._perf_num_cores_slider = None
        self._perf_num_warmups_slider = None
        self._perf_num_test_slider = None
        self._perf_message = None
        self._perf_button = None

    def create(self):
        self._title = widgets.HTML(value='<h3>Model Analysis Helper</h3>')
        self._create_desc_container()
        self._create_sens_container()
        self._create_perf_container()

        return widgets.VBox((
            self._title,
            self._desc_container,
            self._sens_container,
            self._perf_container
        ))

    def _create_desc_container(self):
        self._desc_message = widgets.HTML(value='')
        self._desc_button = widgets.Button(description='Run', button_style='info')
        self._desc_container = widgets.VBox((
            widgets.HTML(value='<h4>Description and ONNX Export</h4>'),
            self._desc_message,
            self._desc_button
        ))
        self._setup_desc()

    def _create_sens_container(self):
        devices = [*['cuda:{}'.format(dev) for dev in range(torch.cuda.device_count()) if torch.cuda.is_available()],
                   'cpu']
        self._sens_device_buttons = {
            device: widgets.ToggleButton(value=False, description=device) for device in devices
        }
        self._sens_device_buttons[devices[0]].value = True
        self._sens_device_buttons_container = widgets.HBox((
            widgets.Label(value='Device'),
            widgets.VBox(tuple([val for (_, val) in self._sens_device_buttons.items()]))
        ))
        dataset_size = len(self.settings.dataset)
        self._sens_batch_size_slider = widgets.FloatLogSlider(value=64, base=2, min=0, max=6,
                                                              step=1, description='Batch Size:')
        self._sens_samples_size_slider = widgets.FloatLogSlider(value=min(64, dataset_size), base=2, min=0,
                                                                max=math.log2(dataset_size), step=1,
                                                                description='Num Samples:')
        self._sens_cache_data_check = widgets.Checkbox(value=True, description='Cache dataset', disabled=False)
        self._sens_message = widgets.HTML(value='')
        self._sens_button = widgets.Button(description='Run', button_style='info')
        self._sens_container = widgets.VBox((
            widgets.HTML(value='<h4>One Shot Sensitivity Analysis</h4>'),
            self._sens_device_buttons_container,
            self._sens_batch_size_slider,
            self._sens_samples_size_slider,
            self._sens_cache_data_check,
            self._sens_message,
            self._sens_button
        ))
        self._setup_sens()

    def _create_perf_container(self):
        self._perf_batch_size_slider = widgets.FloatLogSlider(value=1, base=2, min=0, max=20,
                                                              step=1, description='Batch Size:')
        num_cores = CPUCoresCounter.factory().get_physical_cores_count()
        self._perf_num_cores_slider = widgets.IntSlider(value=num_cores, min=1, max=num_cores, step=1,
                                                        description='Num Cores:')
        self._perf_num_warmups_slider = widgets.FloatLogSlider(value=10, base=2, min=0, max=20,
                                                               description='Num Warmups')
        self._perf_num_test_slider = widgets.FloatLogSlider(value=50, base=2, min=0, max=20,
                                                            description='Num Tests')
        self._perf_message = widgets.HTML(value='')
        self._perf_button = widgets.Button(description='Run', button_style='info')
        self._perf_container = widgets.VBox((
            widgets.HTML(value='<h4>NM Performance Analysis</h4>'),
            self._perf_batch_size_slider,
            self._perf_num_cores_slider,
            self._perf_num_warmups_slider,
            self._perf_num_test_slider,
            self._perf_message,
            self._perf_button
        ))
        self._setup_perf()

    def _setup_desc(self):
        def _run_click(change):
            self._update_desc_message('exporting...')

            try:
                self._export_description()
                self._export_onnx()
                self._update_desc_message('exported!', success=True)
            except Exception as err:
                self._update_desc_message(str(err), error=True)

        self._desc_button.on_click(_run_click)
        self._update_desc_message('Click run to export!')

    def _setup_sens(self):
        def _dev_button_updater(selected: bool, device: str):
            if selected and device == 'cpu':
                for _key, _val in self._sens_device_buttons.items():
                    if _key != 'cpu':
                        _val.value = False
            elif selected:
                for _key, _val in self._sens_device_buttons.items():
                    if _key == 'cpu':
                        _val.value = False

        def _dev_button_trigger(change):
            _dev_button_updater(change['new'], change['owner'].description)

        for key, val in self._sens_device_buttons.items():
            val.observe(_dev_button_trigger, names='value')

        def _batch_size_updater(num_samples: float):
            self._sens_batch_size_slider.max = math.log2(num_samples)
            self._sens_batch_size_slider.value = min(num_samples, self._sens_batch_size_slider.value)

        def _num_samples_trigger(change):
            _batch_size_updater(change['new'])

        self._sens_samples_size_slider.observe(_num_samples_trigger, names='value')

        def _run_click(change):
            self._update_sens_message('analyzing...')

            try:
                self._run_sens_analysis()
                self._update_sens_message('analyzed!', success=True)
            except Exception as err:
                self._update_sens_message(str(err), error=True)

        self._sens_button.on_click(_run_click)

        _batch_size_updater(self._sens_samples_size_slider.value)
        self._update_sens_message('Click run to analyze!')

    def _setup_perf(self):
        pass
        
    def _update_desc_message(self, message: str, success: bool = False, error: bool = False):
        color = ModelAnalysisWidget._get_color(success, error)
        self._desc_message.value = '<span style="color: {}">{}<span>'.format(color, message)

    def _update_sens_message(self, message: str, success: bool = False, error: bool = False):
        color = ModelAnalysisWidget._get_color(success, error)
        self._sens_message.value = '<span style="color: {}">{}<span>'.format(color, message)

    def _update_perf_message(self, message: str, success: bool = False, error: bool = False):
        color = ModelAnalysisWidget._get_color(success, error)
        self._sens_message.value = '<span style="color: {}">{}<span>'.format(color, message)

    def _export_description(self):
        path = os.path.join(self.settings.save_dir, self.settings.save_desc_name)
        path = clean_path(path)
        create_parent_dirs(path)

        self._update_desc_message('\nanalyzing model...')
        analyzer = ModuleAnalyzer(self.settings.model, enabled=True)
        self.settings.model(self.settings.sample_input)
        analyzer.enabled = False
        self._update_desc_message('\nfinished analyzing model...')
        save_model_ks_desc(analyzer, path)
        print('exported model description to {}'.format(path))

    def _export_onnx(self):
        path = os.path.join(self.settings.save_dir, self.settings.save_onnx_name)
        path = clean_path(path)
        create_parent_dirs(path)

        self._update_desc_message('\nexporting onnx...')
        torch.onnx.export(self.settings.model, self.settings.sample_input, path, strip_doc_string=True, verbose=False)
        print('exported onnx to {}'.format(path))

    def _run_sens_analysis(self):
        path = os.path.join(self.settings.save_dir, self.settings.save_sens_name)
        path = clean_path(path)
        create_parent_dirs(path)

        devices = [key for key, val in self._sens_device_buttons.items() if val.value]
        device = devices[0] if len(devices) < 2 else 'cuda:{}'.format(','.join(device[-1] for device in devices))
        batch_size = int(self._sens_batch_size_slider.value)
        samples_per_check = int(self._sens_samples_size_slider.value)
        cache = self._sens_cache_data_check.value

        self._update_sens_message('\nrunning sensitivity analysis...')

        def _progress_callback(_layer_index, _num_layers, _layer_name, _sparsity_levels, _level_index):
            if _level_index == -1:
                self._update_sens_message('running sensitivity analysis for layer ({}/{}) {} and sparsity levels {}'
                                          .format(_layer_index, _num_layers, _layer_name, _sparsity_levels))

        sensitivities = one_shot_sensitivity_analysis(
            self.settings.model, self.settings.dataset, self.settings.loss_fn, device, batch_size, samples_per_check,
            self.settings.one_shot_sparsity_check_levels, cache, self.settings.one_shot_loader_args,
            self.settings.one_shot_data_loader_const, self.settings.one_shot_tester_run_funcs,
            progress_callback=_progress_callback
        )
        save_one_shot_sensitivity_analysis(sensitivities, path)
        print('saved sensitivity analysis to {}'.format(path))

    @staticmethod
    def _get_color(success: bool, error: bool):
        if error:
            return 'red'

        if success:
            return 'green'

        return 'black'
