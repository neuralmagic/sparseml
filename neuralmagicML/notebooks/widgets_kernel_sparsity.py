from typing import Tuple, Union
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
                           init_start_sparsity: int = 0.05, init_final_sparsity: int = 0.5,
                           disable_first_last: bool = True, disable_all: bool = True,
                           init_start_epoch: int = 0.0, init_end_epoch: int = 10.0,
                           init_update_frequency: int = 1.0):
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
        init_widg, init_mod = KSModifierWidgets.interactive_group_module(
            module, flops_analyzer if ran_flops else None, init_start_sparsity, init_final_sparsity,
            disable_first_last, disable_all, init_start_epoch, init_end_epoch, init_update_frequency
        )
        modifiers.append(init_mod)
        tabs.children = (init_widg,)
        tab_counter = 0
        tabs.set_title(0, 'Group {}'.format(tab_counter))

        def _add_new(_state):
            nonlocal tab_counter

            add_widg, add_mod = KSModifierWidgets.interactive_group_module(
                module, flops_analyzer if ran_flops else None, init_start_sparsity, init_final_sparsity,
                True, True, init_start_epoch, init_end_epoch, init_update_frequency
            )
            tabs.children = tuple([*tabs.children, add_widg])
            modifiers.append(add_mod)
            tab_counter += 1
            tabs.set_title(len(tabs.children) - 1, 'Group {}'.format(tab_counter))

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

        return widg_container, modifiers

    @staticmethod
    def interactive_group_module(module: Module, flops_analyzer: Union[FlopsAnalyzerModule, None] = None,
                                 init_start_sparsity: int = 0.05, init_final_sparsity: int = 0.5,
                                 disable_first_last: bool = True, disable_all: bool = True,
                                 init_start_epoch: int = 0.0, init_end_epoch: int = 10.0,
                                 init_update_frequency: int = 1.0) -> Tuple[widgets.Box, GradualKSModifier]:
        """
        ----------------------------------------------------------------------------
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

        """
        total_flops = flops_analyzer.total_flops if flops_analyzer is not None else 0
        total_params = flops_analyzer.total_params if flops_analyzer is not None else 0
        modifier = GradualKSModifier(
            'weight', layers=[], init_sparsity=init_start_sparsity, final_sparsity=init_final_sparsity,
            inter_func='cubic', start_epoch=init_start_epoch, end_epoch=init_end_epoch,
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

            enable_checkbox = widgets.Checkbox(value=not disable_all, description=name)
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

            if not disable_all:
                modifier.layers.append(name)

        def _update_layers():
            for lay_widget, lay_name in zip(lay_widgets, lay_names):
                lay_widget.children[0].value = lay_name in modifier.layers

        def _bulk_enable_change(_change):
            modifier.layers = []

            if _change['new']:
                modifier.layers.extend(lay_names)

            _update_layers()

        bulk_enable_checkbox = widgets.Checkbox(value=not (disable_all or disable_first_last),
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

        if disable_first_last and not disable_all:
            modifier.layers.pop(0)
            modifier.layers.pop()
            _update_from_modifier()

        widg_container = widgets.Box((container,), layout=widgets.Layout(margin='8px'))
        widg_container.update_from_modifier = _update_from_modifier

        return widg_container, modifier
