"""
Code related to modifiers for enforcing activation sparsity on models while training
"""

from typing import Union, List, Tuple
import yaml
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as TF
from torch.optim.optimizer import Optimizer

from neuralmagicML.pytorch.utils import (
    ALL_TOKEN,
    validate_str_list,
    get_terminal_layers,
    get_layer,
    convert_to_bool,
)
from neuralmagicML.pytorch.recal.modifier import ScheduledModifier
from neuralmagicML.pytorch.recal.activation.tracker import ASLayerTracker


__all__ = ["ASRegModifier", "REG_FUNCTIONS", "REG_TENSORS"]


REG_FUNCTIONS = ["l1", "l2", "relu", "hs"]
REG_TENSORS = ["inp", "out"]


class ASRegModifier(ScheduledModifier):
    """
    Add a regularizer over the inputs or outputs to given layers (activation regularization)
    This promotes larger activation sparsity values

    Sample yaml:
        !ASRegModifier
            start_epoch: 0.0
            end_epoch: 10.0
            layers:
                - layer1
                -layer2
            alpha: 0.00001
            layer_normalized: True
            reg_func: l1
            reg_tens: inp
    """

    YAML_KEY = u"!ASRegModifier"

    @staticmethod
    def yaml_constructor(loader, node):
        instance = ASRegModifier.__new__(ASRegModifier)
        yield instance
        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    def __init__(
        self,
        layers: Union[str, List[str]],
        alpha: Union[float, List[float]],
        layer_normalized: bool = False,
        reg_func: str = "l1",
        reg_tens: str = "inp",
        start_epoch: float = -1.0,
        end_epoch: float = -1.0,
    ):
        """
        :param layers: str or list of str for the layers to apply the AS modifier to
                       can also use the token __ALL__ to specify all layers
        :param alpha: the weight to use for the regularization, ie cost = loss + alpha * reg
        :param layer_normalized: True to normalize the values by 1 / L where L is the number of layers
        :param reg_func: the regularization function to apply to the activations, one of: l1, l2, relu, hs
        :param reg_tens: the regularization tensor to apply a function to, one of: inp, out
        :param start_epoch: The epoch to start the modifier at
        :param end_epoch: The epoch to end the modifier at
        """
        super().__init__(start_epoch, end_epoch)
        self._layers = validate_str_list(
            layers, "{} for layers".format(self.__class__.__name__)
        )
        self._alpha = alpha
        self._layer_normalized = convert_to_bool(layer_normalized)
        self._reg_func = reg_func
        self._reg_tens = reg_tens
        self._trackers = []  # type: List[ASLayerTracker]

        if not isinstance(self._alpha, float) and self._layers == ALL_TOKEN:
            raise TypeError(
                "list of alphas {} is not supported with {} for layers in {}".format(
                    self._alpha, ALL_TOKEN, self.__class__.__name__
                )
            )

        if not isinstance(self._alpha, float) and len(self._alpha) != len(self._layers):
            raise ValueError(
                "len(alphas) of {} must match len(layers) of {} in {}".format(
                    len(self._alpha), len(self._layers), self.__class__.__name__
                )
            )

        if self._reg_func not in REG_FUNCTIONS:
            raise ValueError(
                "{} is not a supported reg_func, available are {} for {}".format(
                    self._reg_func, REG_FUNCTIONS, self.__class__.__name__
                )
            )

        if self._reg_tens not in REG_TENSORS:
            raise ValueError(
                "{} is not a supported reg_tens, available are {} for {}".format(
                    self._reg_tens, REG_TENSORS, self.__class__.__name__
                )
            )

    def __del__(self):
        self._trackers.clear()

    @property
    def layers(self) -> Union[str, List[str]]:
        """
        :return: str or list of str for the layers to apply the AS modifier to
                 can also use the token __ALL__ to specify all layers
        """
        return self._layers

    @layers.setter
    def layers(self, value: Union[str, List[str]]):
        """
        :param value: str or list of str for the layers to apply the AS modifier to
                      can also use the token __ALL__ to specify all layers
        """
        self.prop_set_check("layers")
        self._layers = value

    @property
    def alpha(self) -> Union[float, List[float]]:
        """
        :return: the weight to use for the regularization, ie cost = loss + alpha * reg
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: Union[float, List[float]]):
        """
        :param value: the weight to use for the regularization, ie cost = loss + alpha * reg
        """
        self.prop_set_check("alpha")
        self._alpha = value

    @property
    def layer_normalized(self):
        return self._layer_normalized

    @layer_normalized.setter
    def layer_normalized(self, value):
        self.prop_set_check("layer_normalized")
        self._layer_normalized = value

    @property
    def reg_func(self) -> str:
        """
        :return: the regularization function to apply to the activations, one of: l1, l2, relu, hs
        """
        return self._reg_func

    @reg_func.setter
    def reg_func(self, value: str):
        """
        :param value: the regularization function to apply to the activations, one of: l1, l2, relu, hs
        """
        self.prop_set_check("reg_func")
        self._reg_func = value

    @property
    def reg_tens(self) -> str:
        """
        :return: the regularization tensor to apply a function to, one of: inp, out
        """
        return self._reg_tens

    @reg_tens.setter
    def reg_tens(self, value: str):
        """
        :param value: the regularization tensor to apply a function to, one of: inp, out
        """
        self.prop_set_check("reg_tens")
        self._reg_tens = value

    def initialize(self, module: Module, optimizer: Optimizer):
        """
        Grab the layer's to control activation sparsity for

        :param module: module to modify
        :param optimizer: optimizer to modify
        """
        super(ASRegModifier, self).initialize(module, optimizer)
        layers = (
            get_terminal_layers(module)
            if self._layers == ALL_TOKEN
            else [get_layer(name, module) for name in self._layers]
        )

        for layer in layers:
            self._trackers.append(
                ASLayerTracker(
                    layer,
                    track_input=self._reg_tens == "inp",
                    track_output=self._reg_tens == "out",
                    input_func=self._regularize_tracked,
                    output_func=self._regularize_tracked,
                )
            )

    def update(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Update the loss tracking for each layer that is being modified on start and stop

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        super().update(module, optimizer, epoch, steps_per_epoch)

        if self.start_pending(epoch, steps_per_epoch):
            for tracker in self._trackers:
                tracker.enable()

        if self.end_pending(epoch, steps_per_epoch):
            for tracker in self._trackers:
                tracker.disable()

    def loss_update(
        self,
        loss: Tensor,
        module: Module,
        optimizer: Optimizer,
        epoch: float,
        steps_per_epoch: int,
    ) -> Tensor:
        """
        modify the loss to include the norms for the outputs of the layers being modified

        :param loss: The calculated loss tensor
        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        :return: the modified loss tensor
        """
        loss = super().loss_update(loss, module, optimizer, epoch, steps_per_epoch)

        act_reg = 0.0
        key = "cpu" if not loss.is_cuda else "cuda:{}".format(loss.get_device())

        for index, tracker in enumerate(self._trackers):
            if self._reg_tens == "inp":
                tracker_reg = tracker.tracked_input[key]
            elif self._reg_tens == "out":
                tracker_reg = tracker.tracked_output[key]
            else:
                raise ValueError(
                    "unsupported reg_tens given of {}".format(self._reg_tens)
                )

            alpha = (
                self._alpha if isinstance(self._alpha, float) else self._alpha[index]
            )
            act_reg += tracker_reg * alpha

        if self._layer_normalized:
            # normalize across the number of layers we are tracking
            act_reg = act_reg / len(self._trackers)

        return loss + act_reg

    def optimizer_post_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        be sure to clear out the values after the update step has been taken

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch (calculate batch number using this and epoch)
        """
        super().optimizer_post_step(module, optimizer, epoch, steps_per_epoch)

        for tracker in self._trackers:
            tracker.clear()

    def _regularize_tracked(self, tens: Union[Tuple[Tensor, ...], Tensor]):
        if isinstance(tens, Tensor):
            tens = (tens,)

        reduced = 0.0

        for ten in tens:
            if self._reg_func == "l1":
                tens_reduced = ten.abs().sum()
            elif self._reg_func == "l2":
                tens_reduced = ten.pow(2).sum()
            elif self._reg_func == "relu":
                tens_reduced = TF.relu(reduced).sum()
            elif self._reg_func == "hs":
                tens_reduced = ten.abs().sum().pow(2) / ten.pow(2).sum()
            else:
                raise ValueError(
                    "unsupported reg_func given of {}".format(self._reg_func)
                )

            # normalize by batch size
            reduced += tens_reduced / ten.shape[0]

        # normalize across all the tensors that were inputs or outputs for the layer
        reduced = reduced / len(tens)

        return reduced


yaml.add_constructor(ASRegModifier.YAML_KEY, ASRegModifier.yaml_constructor)
yaml.add_constructor(
    ASRegModifier.YAML_KEY, ASRegModifier.yaml_constructor, yaml.SafeLoader
)
