from typing import Union, Tuple, List
import os
import numpy
import pandas
import matplotlib.pyplot as plt
from torch import Tensor
from torch.nn import Module

from ..module_analyzer import ModuleAnalyzer, AnalyzedLayerDesc
from ..utils import get_conv_layers, get_linear_layers


__all__ = ['plot_sensitivities', 'ApproxLayerSensitivity', 'approx_sensitivity_analysis']


def plot_sensitivities(sens_vals: List[Tuple[str, float]], title: str = None, normalize: bool = True,
                       save_path: str = None):
    layers, values = zip(*sens_vals)

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
    frame = pandas.DataFrame(list(zip(layers, values)), columns=['Layer', 'Sensitivity'])
    frame.plot.barh(ax=ax, x='Layer', y='Sensitivity')
    plt.gca().invert_yaxis()

    if save_path is None:
        plt.show()
    else:
        save_path = os.path.abspath(os.path.expanduser(save_path))
        save_dir = os.path.pardir(save_path)

        if not os.path.exists(save_dir) and save_dir:
            os.makedirs(save_dir)

        plt.savefig(save_path)
        plt.close(fig)


class ApproxLayerSensitivity(object):
    def __init__(self, layer_desc: AnalyzedLayerDesc):
        self._layer_desc = layer_desc

    @property
    def layer_desc(self) -> AnalyzedLayerDesc:
        return self._layer_desc

    @property
    def uniform(self) -> float:
        return 1.0

    @property
    def er(self) -> float:
        vals_sum = float(self._layer_desc.in_channels + self._layer_desc.out_channels)
        vals_prod = float(self._layer_desc.in_channels * self._layer_desc.out_channels)

        return vals_sum / vals_prod

    @property
    def erk(self) -> float:
        vals_sum = float(self._layer_desc.in_channels + self._layer_desc.out_channels +
                         sum(self._layer_desc.kernel_size))
        vals_prod = float(self._layer_desc.in_channels * self._layer_desc.out_channels *
                          numpy.prod(self._layer_desc.kernel_size))

        return vals_sum / vals_prod

    @property
    def vs_er(self) -> float:
        vol_change = self._volume_change()

        return vol_change * self.er

    @property
    def vs_erk(self) -> float:
        vol_change = self._volume_change()

        return vol_change * self.erk

    @property
    def vs_kernels(self) -> float:
        vol_change = self._volume_change()
        kernels = sum(self._layer_desc.kernel_size) / numpy.prod(self._layer_desc.kernel_size)

        return vol_change * kernels

    def _volume_change(self) -> float:
        inp_vol = numpy.prod(self._layer_desc.input_shape[0][1:]) if self._layer_desc.input_shape else 1.0
        out_vol = numpy.prod(self._layer_desc.output_shape[0][1:]) if self._layer_desc.output_shape else 1.0

        return 1.0 + (abs(inp_vol - out_vol) / ((inp_vol + out_vol) / 2.0))


def approx_sensitivity_analysis(model: Module, inp: Union[Tuple[Tensor, ...], Tensor]) -> List[ApproxLayerSensitivity]:
    analyzer = ModuleAnalyzer(model)
    analyzer.enabled = True
    model(inp)
    analyzer.enabled = False

    sensitivities = []
    conv_layers = get_conv_layers(model)
    linear_layers = get_linear_layers(model)

    for name, _ in conv_layers.items():
        desc = analyzer.layer_desc(name)
        sensitivities.append(ApproxLayerSensitivity(desc))

    for name, _ in linear_layers.items():
        desc = analyzer.layer_desc(name)
        sensitivities.append(ApproxLayerSensitivity(desc))

    sensitivities.sort(key=lambda val: val.layer_desc.call_order)

    return sensitivities


def static_sensitivity_analysis(model: Module, ):
    pass
