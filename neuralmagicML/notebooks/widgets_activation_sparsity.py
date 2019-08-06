from typing import Tuple, Union, List
from collections import OrderedDict
import ipywidgets as widgets
import torch
from torch.nn import (
    Module, Linear, Softmax, Softmax2d,
    Threshold, ReLU, ReLU6, RReLU, LeakyReLU, PReLU, ELU, CELU, SELU, GLU,
    Hardtanh, Tanh, Sigmoid, LogSigmoid
)
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd
from torch.nn.modules.pooling import _AdaptiveMaxPoolNd, _AdaptiveAvgPoolNd

from ..sparsity.activation.analyzer import ASAnalyzerLayer, ASAnalyzerModule
from ..utils.flops_analyzer import FlopsAnalyzerModule


__all__ = ['ASAnalyzerWidgets']


class ASAnalyzerWidgets(object):
    INPUTS_SAMPLE_SIZE = 1000
    OUTPUTS_SAMPLE_SIZE = 1000

    @staticmethod
    def interactive_module(module: Module, device: str = 'cpu', inp_dim: Union[None, Tuple[int, ...]] = None) \
            -> Tuple[widgets.Accordion, List[ASAnalyzerLayer]]:
        flops_analyzer = FlopsAnalyzerModule(module).to(device)
        ran_flops = False

        if inp_dim is not None:
            flops_analyzer.eval()

            with torch.no_grad():
                inp = torch.randn(*inp_dim).to(device)
                flops_analyzer(inp)

            ran_flops = True

        groups = OrderedDict()

        for name, mod in module.named_modules():
            if isinstance(mod, _ConvNd):
                if 'Convs' not in groups:
                    groups['Convs'] = []

                groups['Convs'].append((name, mod))
            elif isinstance(mod, Linear):
                if 'FCs' not in groups:
                    groups['FCs'] = []

                groups['FCs'].append((name, mod))
            elif isinstance(mod, _BatchNorm):
                if 'BNs' not in groups:
                    groups['BNs'] = []

                groups['BNs'].append((name, mod))
            elif isinstance(mod, _MaxPoolNd) or isinstance(mod, _AvgPoolNd) or \
                    isinstance(mod, _AdaptiveAvgPoolNd) or isinstance(mod, _AdaptiveMaxPoolNd):
                if 'Pools' not in groups:
                    groups['Pools'] = []

                groups['Pools'].append((name, mod))
            elif (isinstance(mod, Threshold) or isinstance(mod, ReLU) or isinstance(mod, ReLU6) or
                  isinstance(mod, RReLU) or isinstance(mod, LeakyReLU) or isinstance(mod, PReLU) or
                  isinstance(mod, ELU) or isinstance(mod, CELU) or isinstance(mod, SELU) or isinstance(mod, GLU) or
                  isinstance(mod, Hardtanh) or isinstance(mod, Tanh) or isinstance(mod, Sigmoid) or
                  isinstance(mod, LogSigmoid)) or isinstance(mod, Softmax) or isinstance(mod, Softmax2d):
                if 'Acts' not in groups:
                    groups['Acts'] = []

                groups['Acts'].append((name, mod))
            else:
                if len([m for m in module.modules()]) >= 2:
                    continue

                if 'Others' not in groups:
                    groups['Others'] = []

                groups['Others'].append((name, mod))

        layer_analyzers = []
        group_widgets = []

        for group_name, group_mods in groups.items():
            group_mods = [(name, mod, flops_analyzer.layer_flops(name) if ran_flops else 0,
                           flops_analyzer.layer_params(name) if ran_flops else 0) for (name, mod) in group_mods]
            widg, analyz = ASAnalyzerWidgets.interactive_module_group(group_mods,
                                                                      flops_analyzer.total_flops if ran_flops else 0,
                                                                      flops_analyzer.total_params if ran_flops else 0)
            layer_analyzers.extend(analyz)
            group_widgets.append(widg)

        accordion = widgets.Accordion(children=group_widgets)

        for group_index, group_name in enumerate(groups.keys()):
            accordion.set_title(group_index, group_name)

        del flops_analyzer

        return accordion, layer_analyzers

    @staticmethod
    def interactive_module_group(layers: List[Tuple[str, Module, int, int]],
                                 total_flops: int = 0, total_params: int = 0) -> Tuple[widgets.Box,
                                                                                       List[ASAnalyzerLayer]]:
        """
        ----------------------------------------------------------------------------
        |         [] toggle all input sparsity                                     |
        |         [] toggle all input distribution                                 |
        |         [] toggle all output sparsity                                    |
        |         [] toggle all output distribution                                |
        |                                                                          |
        |   Name: ...             | [] input sparsity   [] input distribution      |
        |   Info: ...             |                                                |
        |   Calc: ...             | [] output sparsity  [] output distribution     |
        |                                                                          |
        |                                                                          |
        |   Name: ...             | [] input sparsity   [] input distribution      |
        |   Info: ...             |                                                |
        |   Calc: ...             | [] output sparsity  [] output distribution     |
        |                                                                          |
        ----------------------------------------------------------------------------

        :param layers: the list of layers to plot of format
                       (name of layer, layer module, flops for layer, params in layer)
        :param total_flops: the total flops in the parent module
        :param total_params: the total params in the parent module
        :return: a tuple of the widget to display and a list of the analyzer layers the widget controls
        """
        lay_widgets = []
        lay_analyzers = []

        for lay_info in layers:
            name = lay_info[0]
            layer = lay_info[1]
            flops = lay_info[2]
            params = lay_info[3]

            widget, analyzer = ASAnalyzerWidgets.interactive_module_layer(name, layer, flops, total_flops,
                                                                          params, total_params)
            lay_widgets.append(widget)
            lay_analyzers.append(analyzer)

        input_sparsity_check = widgets.Checkbox(value=False, description='toggle all input sparsity')
        input_distribution_check = widgets.Checkbox(value=False, description='toggle all input distribution')
        output_sparsity_check = widgets.Checkbox(value=False, description='toggle all output sparsity')
        output_distribution_check = widgets.Checkbox(value=False, description='toggle all output distribution')

        def _input_sparsity_change(_change):
            if _change['type'] != 'change' or not isinstance(_change['new'], bool):
                return

            for lay_widget in lay_widgets[1:]:
                checkbox = lay_widget.children[0].children[1].children[0]
                checkbox.value = _change['new']

        def _input_distribution_change(_change):
            if _change['type'] != 'change' or not isinstance(_change['new'], bool):
                return

            for lay_widget in lay_widgets[1:]:
                checkbox = lay_widget.children[0].children[1].children[1]
                checkbox.value = _change['new']

        def _output_sparsity_change(_change):
            if _change['type'] != 'change' or not isinstance(_change['new'], bool):
                return

            for lay_widget in lay_widgets[1:]:
                checkbox = lay_widget.children[0].children[1].children[2]
                checkbox.value = _change['new']

        def _output_distribution_change(_change):
            if _change['type'] != 'change' or not isinstance(_change['new'], bool):
                return

            for lay_widget in lay_widgets[1:]:
                checkbox = lay_widget.children[0].children[1].children[3]
                checkbox.value = _change['new']

        input_sparsity_check.observe(_input_sparsity_change)
        input_distribution_check.observe(_input_distribution_change)
        output_sparsity_check.observe(_output_sparsity_change)
        output_distribution_check.observe(_output_distribution_change)
        all_box = widgets.VBox((
            input_sparsity_check, input_distribution_check,
            output_sparsity_check, output_distribution_check
        ))
        lay_widgets.insert(0, all_box)
        container = widgets.VBox(tuple(lay_widgets))

        return widgets.Box((container,)), lay_analyzers

    @staticmethod
    def interactive_module_layer(name: str, layer: Module, flops: int = 0, total_flops: int = 0,
                                 params: int = 0, total_params: int = 0) -> Tuple[widgets.Box, ASAnalyzerLayer]:
        """
        --------------------------------------------------------------------------
        | Name: ...             | [] input sparsity   [] input distribution      |
        | Info: ...             |                                                |
        | Calc: ...             | [] output sparsity  [] output distribution     |
        --------------------------------------------------------------------------

        :param name: name of layer in the parent module / model
        :param layer: the layer to create an AS analyzer for
        :param flops: the number of flops needed to run the layer
        :param total_flops: the number of flops needed to run the parent module / model
        :param params: the number of params in the layer
        :param total_params: the number of params in the parent module / model
        :return: a tuple of the widget to display and the analyzer layer the widget controls
        """
        analyzer = ASAnalyzerLayer(name, division=0, enabled=False)

        input_sparsity_check = widgets.Checkbox(value=False, description='input sparsity')
        input_distribution_check = widgets.Checkbox(value=False, description='input distribution')
        output_sparsity_check = widgets.Checkbox(value=False, description='output sparsity')
        output_distribution_check = widgets.Checkbox(value=False, description='output distribution')

        def _fix_analyzer_state():
            enable = (analyzer.track_inputs_sparsity or analyzer.inputs_sample_size > 0 or
                      analyzer.track_outputs_sparsity or analyzer.outputs_sample_size > 0)

            if enable:
                analyzer.enable()
            else:
                analyzer.disable()

        def _input_sparsity_change(_change):
            if _change['type'] != 'change' or not isinstance(_change['new'], bool):
                return

            analyzer.track_inputs_sparsity = _change['new']
            _fix_analyzer_state()

        def _input_distribution_change(_change):
            if _change['type'] != 'change' or not isinstance(_change['new'], bool):
                return

            analyzer.inputs_sample_size = ASAnalyzerWidgets.INPUTS_SAMPLE_SIZE if _change['new'] else 0
            _fix_analyzer_state()

        def _output_sparsity_change(_change):
            if _change['type'] != 'change' or not isinstance(_change['new'], bool):
                return

            analyzer.track_outputs_sparsity = _change['new']
            _fix_analyzer_state()

        def _output_distribution_change(_change):
            if _change['type'] != 'change' or not isinstance(_change['new'], bool):
                return

            analyzer.outputs_sample_size = ASAnalyzerWidgets.OUTPUTS_SAMPLE_SIZE if _change['new'] else 0
            _fix_analyzer_state()

        input_sparsity_check.observe(_input_sparsity_change)
        input_distribution_check.observe(_input_distribution_change)
        output_sparsity_check.observe(_output_sparsity_change)
        output_distribution_check.observe(_output_distribution_change)

        checks_box = widgets.GridBox((
            input_sparsity_check, input_distribution_check, output_sparsity_check, output_distribution_check
        ), layout=widgets.Layout(grid_template_rows='auto auto', grid_template_columns='25% 25%'))

        description = '{}: '.format(layer.__class__.__name__)

        if isinstance(layer, _ConvNd):
            description += 'i{}; o{}; k{}; p{}; s{}; b{}'.format(layer.in_channels, layer.out_channels,
                                                                 layer.kernel_size, layer.padding, layer.stride,
                                                                 layer.bias is not None)
        elif isinstance(layer, Linear):
            description += 'i{}; o{}; b{}'.format(layer.in_features, layer.out_features,
                                                  layer.bias is not None)
        elif isinstance(layer, _BatchNorm):
            description += 'c{}'.format(layer.num_features)
        elif isinstance(layer, _MaxPoolNd) or isinstance(layer, _AvgPoolNd):
            layer = layer  # type: _MaxPoolNd
            description += 'k{}; p{}; s{}'.format(layer.kernel_size, layer.padding, layer.stride)

        text_box = widgets.VBox((
            widgets.HTML(value='Name: <b>{}</b>'.format(name)),
            widgets.HTML(value='Info: <i>{}</i>'.format(description)),
            widgets.HTML(value='Calc: <i>{} flops ({:.2f}%), {} params ({:.2f}%)</i>'
                         .format(flops, 100.0 * flops / total_flops if total_flops > 0 else 'NA',
                                 params, 100.0 * params / total_params if total_params > 0 else 'NA'))
        ))
        container = widgets.HBox((text_box, checks_box))

        return widgets.Box((container,), layout=widgets.Layout(margin='8px')), analyzer
