from typing import List, Union, Tuple
from torch import Tensor
from torch.nn import Module, Parameter

from ..utils import tensor_sparsity


__all__ = ['KSAnalyzerLayer']


class KSAnalyzerLayer(object):
    @staticmethod
    def analyze_layers(module: Module, layers: List[str], param_name: str = 'weight'):
        analyzed = []

        for layer_name in layers:
            mod = module
            lays = layer_name.split('.')

            for lay in lays:
                mod = mod.__getattr__(lay)

            analyzed.append(KSAnalyzerLayer(mod, layer_name, param_name))

        return analyzed

    def __init__(self, layer: Module, name: str, param_name: str = 'weight'):
        """
        Analyzer to get the sparsity of a given layer's parameter such as weight

        :param layer: the layer containing the param to analyze the sparsity for
        :param name: name of the layer, used for tracking
        :param param_name: name of the parameter to analyze the sparsity for, defaults to weight
        """
        self._layer = layer
        self._name = name
        self._param_name = param_name
        self._param = self._layer.__getattr__(self._param_name)  # type: Parameter

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def name(self) -> str:
        return self._name

    @property
    def param_name(self) -> str:
        return self._param_name

    @property
    def tag(self):
        return '{}.{}'.format(self.name, self.param_name)

    @property
    def param(self) -> Parameter:
        return self._param

    @property
    def param_sparsity(self) -> Tensor:
        return self.param_sparsity_division(None)

    def param_sparsity_division(self, division: Union[None, int, Tuple[int, ...]] = None) -> Tensor:
        return tensor_sparsity(self._param.data, division)
