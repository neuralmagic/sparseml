from typing import Tuple, Union, Dict, List
import ipywidgets as widgets
import torch
from torch.nn import Module, Linear
from torch.nn.modules.conv import _ConvNd

from ..sparsity.kernel.modifier_ks import GradualKSModifier
from ..utils.flops_analyzer import FlopsAnalyzerModule


__all__ = ['KSModifierWidgets']


class KSModifierWidgets(object):
    @staticmethod
    def interactive_module(module: Module, device: str = 'cpu', inp_dim: Union[None, Tuple[int, ...]] = None,
                           **new_group_kwargs: Dict) -> Tuple[widgets.Box, List[GradualKSModifier]]:
        """
        ----------------------------------------------------------------------------
        |                                                                          |
        | [ Delete Current ]  [ Add New ]                                          |
        |  _______   _______   _______                                             |
        | |_______| |_______| |_______|                                            |
        |                                                                          |
        | Sparsity  0===0==== 0.01 - 0.5                                           |
        | Start Epoch [___]                                                        |
        | End Epoch [___]                                                          |
        | Update Freq [___]                                                        |
        |                                                                          |
        | [ Layers ^ ]                                                             |
        |   []  Name     Info: ...                                                 |
        |                Calc: ...                                                 |
        |                                                                          |
        |   []  Name     Info: ...                                                 |
        |                Calc: ...                                                 |
        |                                                                          |
        |   []  Name     Info: ...                                                 |
        |                Calc: ...                                                 |
        |                                                                          |
        ----------------------------------------------------------------------------

        :param module: the module to create ks modifier and widgets for
        :param device: the device to run flops calculation on
        :param inp_dim: the input dimensions for the flops calculator to use, if None will not analyze flops
        :param new_group_kwargs: the kwargs to use when creating a new group using the widget button
        :return: the created widget and the list of KSModifiers, widget has functions add_group, update_from_modifiers
        """
        modifiers = []
        flops_analyzer = FlopsAnalyzerModule(module).to(device)
        ran_flops = False

        if inp_dim is not None:
            flops_analyzer.eval()

            with torch.no_grad():
                inp = torch.randn(*inp_dim).to(device)
                flops_analyzer(inp)

            ran_flops = True

        flops_analyzer.disable()

        tabs = widgets.Tab()
        tab_counter = 0

        def _add_group(init_start_sparsity: int = 0.05, init_final_sparsity: int = 0.85, init_enabled: bool = True,
                       init_start_epoch: int = 0.0, init_end_epoch: int = 30.0, init_update_frequency: int = 1.0,
                       inter_func: str = 'cubic'):
            nonlocal tab_counter

            add_widg, add_mod = KSModifierWidgets.interactive_group_module(
                module, flops_analyzer if ran_flops else None, init_start_sparsity, init_final_sparsity, init_enabled,
                init_start_epoch, init_end_epoch, init_update_frequency, inter_func
            )
            modifiers.insert(0, add_mod)
            tabs.children = tuple([*tabs.children, add_widg])
            tabs.set_title(len(tabs.children) - 1, 'Group {}'.format(tab_counter))
            tab_counter += 1

        def _add_new(_state):
            _add_group(**new_group_kwargs)

        def _delete_current(_state):
            children = [*tabs.children]
            modifiers.pop(tabs.selected_index)
            children.pop(tabs.selected_index)
            tabs.children = children

        add_button = widgets.Button(description='Add New Group')
        delete_button = widgets.Button(description='Delete Current Group')

        delete_button.on_click(_delete_current)
        add_button.on_click(_add_new)

        container = widgets.VBox((
            widgets.HBox((add_button, delete_button)),
            widgets.Box(layout=widgets.Layout(height='16px')),
            tabs
        ))

        def _update_from_modifiers():
            for _widg in tabs.children:
                _widg.update_from_modifier()

        widg_container = widgets.Box((container,), layout=widgets.Layout(margin='16px'))
        widg_container.update_from_modifiers = _update_from_modifiers
        widg_container.add_group = _add_group

        return widg_container, modifiers

    @staticmethod
    def interactive_group_module(
            module: Module, flops_analyzer: Union[FlopsAnalyzerModule, None] = None,
            init_start_sparsity: int = 0.05, init_final_sparsity: int = 0.85, init_enabled: bool = True,
            init_start_epoch: int = 0.0, init_end_epoch: int = 30.0, init_update_frequency: int = 1.0,
            inter_func: str = 'cubic') -> Tuple[widgets.Box, GradualKSModifier]:
        """
        ----------------------------------------------------------------------------
        |                                                                          |
        | Sparsity  0===0==== 0.01 - 0.5                                           |
        | Start Epoch [___]                                                        |
        | End Epoch [___]                                                          |
        | Update Freq [___]                                                        |
        |                                                                          |
        | [ Layers ^ ]                                                             |
        |   []  Name     Info...                                                   |
        |                Calc...                                                   |
        |                                                                          |
        |   []  Name     Info...                                                   |
        |                Calc...                                                   |
        |                                                                          |
        |   []  Name     Info...                                                   |
        |                Calc...                                                   |
        |                                                                          |
        ----------------------------------------------------------------------------

        :param module: the module to create the interactive group for
        :param flops_analyzer: the flops analyzer used to analyze the flops for the module, none if not analyzed
        :param init_start_sparsity: the initial start sparsity for the KSModifier
        :param init_final_sparsity: the initial final sparsity for the KSModifier
        :param init_enabled: True if initially enabled for all layers else False
        :param init_start_epoch: the initial start epoch for the KSModifier
        :param init_end_epoch: the initial end epoch for the KSModifier
        :param init_update_frequency: the initial udpate frequency for the KSModifier
        :param inter_func: the interpolation function to use in the KSModifier
        :return:
        """
        total_flops = flops_analyzer.total_flops if flops_analyzer is not None else 0
        total_params = flops_analyzer.total_params if flops_analyzer is not None else 0
        modifier = GradualKSModifier(
            'weight', layers=[], init_sparsity=init_start_sparsity, final_sparsity=init_final_sparsity,
            inter_func=inter_func, start_epoch=init_start_epoch, end_epoch=init_end_epoch,
            update_frequency=init_update_frequency
        )
        lay_widgets = []
        lay_names = []

        sparsity_slider = widgets.FloatRangeSlider(value=[init_start_sparsity, init_final_sparsity],
                                                   min=0.0, max=1.0, step=0.01, description='Sparsity')
        start_text = widgets.FloatText(value=modifier.start_epoch, description='Start Epoch')
        end_text = widgets.FloatText(value=modifier.end_epoch, description='End Epoch')
        freq_text = widgets.FloatText(value=modifier.update_frequency, description='Update Freq')

        def _sparsity_change(_change):
            modifier.init_sparsity = _change['new'][0]
            modifier.final_sparsity = _change['new'][1]

        def _start_change(_change):
            modifier.start_epoch = _change['new']

        def _end_change(_change):
            modifier.end_epoch = _change['new']

        def _freq_change(_change):
            modifier.update_frequency = _change['new']

        sparsity_slider.observe(_sparsity_change, names='value')
        start_text.observe(_start_change, names='value')
        end_text.observe(_end_change, names='value')
        freq_text.observe(_freq_change, names='value')

        for name, mod in module.named_modules():
            if not isinstance(mod, _ConvNd) and not isinstance(mod, Linear):
                continue

            def _enable_change(_change):
                cur_layers = modifier.layers

                if name in cur_layers and not _change['new']:
                    # remove the layer
                    cur_layers.remove(name)
                elif name not in cur_layers and _change['new']:
                    # add the layer
                    cur_layers.append(name)

            enable_checkbox = widgets.Checkbox(value=init_enabled, description=name)
            enable_checkbox.observe(_enable_change, names='value')
            description = '{}: '.format(mod.__class__.__name__)

            if isinstance(mod, _ConvNd):
                description += ('i{}; o{}; k{}; p{}; s{}; b{}'
                                .format(mod.in_channels, mod.out_channels, mod.kernel_size,
                                        mod.padding, mod.stride, mod.bias is not None))
            elif isinstance(mod, Linear):
                description += 'i{}; o{}; b{}'.format(mod.in_features, mod.out_features, mod.bias is not None)

            mod_flops = flops_analyzer.layer_flops(name) if flops_analyzer is not None else 0
            mod_params = flops_analyzer.layer_params(name) if flops_analyzer is not None else 0

            extras_container = widgets.VBox((
                widgets.HTML(value='<i>{}</i>'.format(description)),
                widgets.HTML(value='<i>{} flops ({:.2f}%), {} params ({:.2f}%)</i>'.format(
                    mod_flops, 100.0 * mod_flops / (total_flops if flops_analyzer is not None else -1.0),
                    mod_params, 100.0 * mod_params / (total_params if flops_analyzer is not None else -1.0))
                )
            ))
            mod_container = widgets.HBox((enable_checkbox, extras_container),
                                         layout=widgets.Layout(margin='4px'))
            lay_widgets.append(mod_container)
            lay_names.append(name)

            if init_enabled:
                modifier.layers.append(name)

        def _update_layers():
            for lay_widget, lay_name in zip(lay_widgets, lay_names):
                lay_widget.children[0].value = lay_name in modifier.layers

        def _bulk_enable_change(_change):
            modifier.layers = []

            if _change['new']:
                modifier.layers.extend(lay_names)

            _update_layers()

        bulk_enable_checkbox = widgets.Checkbox(value=init_enabled,
                                                description='enable / disable all',
                                                layout=widgets.Layout(margin='8px'))
        bulk_enable_checkbox.observe(_bulk_enable_change, names='value')

        layers_accordion = widgets.Accordion((
            widgets.VBox((bulk_enable_checkbox, *lay_widgets)),
        ))
        layers_accordion.set_title(0, 'Selectable Layers')
        container = widgets.VBox((
            widgets.HTML('<b>Layer Pruning Group</b>'),
            sparsity_slider,
            start_text,
            end_text,
            freq_text,
            widgets.Box((), layout=widgets.Layout(height='16px')),
            layers_accordion
        ))

        def _update_from_modifier():
            sparsity_slider.value = (modifier.init_sparsity, modifier.final_sparsity)
            start_text.value = modifier.start_epoch
            end_text.value = modifier.end_epoch
            freq_text.value = modifier.update_frequency
            _update_layers()

        widg_container = widgets.Box((container,), layout=widgets.Layout(margin='8px'))
        widg_container.update_from_modifier = _update_from_modifier

        return widg_container, modifier
