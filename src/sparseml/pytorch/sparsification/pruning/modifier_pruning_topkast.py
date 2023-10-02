# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, OrderedDict, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.mask_creator import (
    PruningMaskCreator,
    get_mask_creator_default,
)
from sparseml.pytorch.sparsification.pruning.mask_params import ModuleParamPruningMask
from sparseml.pytorch.sparsification.pruning.modifier_pruning_base import (
    BasePruningModifier,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_magnitude import (
    MagnitudePruningParamsScorer,
)
from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsScorer


__all__ = ["TopKASTPruningModifier"]


@PyTorchModifierYAML()
class TopKASTPruningModifier(BasePruningModifier):
    """
    Implementation of
    Top-KAST: Top-K Always Sparse Training:
    https://arxiv.org/pdf/2106.03517.pdf.
    Top-KAST uses two masks: a forward mask which is applied during inference and
    the forward pass of training, and a backward mask which is applied to sparsify
    the (otherwise dense) gradient. Both masks are magnitude-based, with the former
    being more restrictive than the latter.
    In addition, unmasked weights are decayed by adding an L2 regularization term
    to the loss.
    Note that in the original paper, top-Kast is used with SGD with NO momentum
    and masks are recomputed every step (though the authors also test mask
    recomputation every 100 steps with no change in accuracy.)
    | Sample yaml:
    |   !TopKASTPruningModifier
    |       forward_sparsity: 0.9
    |       backward_sparsity: 0.5
    |       start_epoch: 0
    |       end_epoch: 100
    |       update_frequency: -1
    |       params: __ALL_PRUNABLE__
    |       global_sparsity: True
    |       acive_param_decay: 0.0001

    :param forward_sparsity: The sparsity of the weights during the forward pass.
    :param backward sparsity: The sparsity of the gradients during the backward pass.
        Should be at least as high as the forward sparsity.
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The length (in epochs) between the forward and backward
        masks are computed. Can and likely should be fractional or -1 to recompute
        at every epoch.
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights. If a sparsity to param mapping is defined by
        final_sparsity, then params should be set to []
    :param global_sparsity: set True to enable global pruning. if False, pruning will
        be layer-wise. Default is True.
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune. Default is True
    :param mask_type: String to define type of sparsity to apply. May be 'unstructred'
        for unstructured pruning or 'block4' for four block pruning or a list of two
        integers for a custom block shape. Default is 'unstructured'
    :param active_weight_decay: How much to decay active (not masked-out) params at
        every step. Note that this value is also multiplied by the learning rate.
        Parameters that are fully active are decayed by active_weight_decay*lr;
        Params that are inactive for the forward pass but updated in the backward
        pass are decayed by active_weight_decay*lr*(1-sparsity).
    """

    def __init__(
        self,
        forward_sparsity: float,
        backward_sparsity: float,
        start_epoch: Union[int, float],
        end_epoch: Union[int, float],
        update_frequency: Union[int, float],
        params: Union[str, List[str]],
        global_sparsity: bool = True,
        leave_enabled: bool = True,
        mask_type: str = "unstructured",
        active_weight_decay: float = 0.0001,
    ):

        self._mask_type = mask_type

        super(TopKASTPruningModifier, self).__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            global_sparsity=global_sparsity,
            params=params,
            leave_enabled=leave_enabled,
            allow_reintroduction=True,
        )

        self._forward_sparsity = forward_sparsity
        self._backward_sparsity = backward_sparsity
        self._active_weight_decay = active_weight_decay

    def initialize_extras(self, module):
        named_layers_and_params = self._create_named_layers_and_params(module)
        layers = [nlp.layer for nlp in named_layers_and_params]
        param_names = [nlp.param_name for nlp in named_layers_and_params]
        layer_names = [nlp.layer_name for nlp in named_layers_and_params]

        # initialize mask_creator and scorer
        params = [
            getattr(layer, param_name) for layer, param_name in zip(layers, param_names)
        ]
        full_param_names = [
            f"{layer_name}.{param_name}"
            for layer_name, param_name in zip(layer_names, param_names)
        ]

        # We  need a whole separate set of masks for gradients,
        # since they will be pruned to a smaller sparsity than weights.
        self._grad_mask_creator = self._get_mask_creator(full_param_names, params)
        self._grad_scorer = self._get_scorer(params)
        self._grad_module_masks = self._create_grad_pruning_mask(
            layers, layer_names, param_names
        )

    def get_applied_sparsity_for_epoch(
        self, epoch: float, steps_per_epoch: int
    ) -> Union[float, List[float]]:
        """
        :param epoch: current epoch
        :param steps_per_epoch: number of steps per epoch
        :return: sparsity level that should be applied at the given epoch. If parameters
            should be set to different sparsities, should return a list of those values
            in the order the parameters appear in the mask manager for this object
        """
        return self._forward_sparsity

    def get_applied_grad_sparsity_for_epoch(
        self, epoch: float, steps_per_epoch: int
    ) -> Union[float, List[float]]:
        """
        :param epoch: current epoch
        :param steps_per_epoch: number of steps per epoch
        :return: **gradient** sparsity level that should be applied at the given
            epoch. If parameters should be set to different sparsities, should
            return a list of those values in the order the parameters appear in
            the mask manager for this object
        """
        return self._backward_sparsity

    @ModifierProp()
    def mask_type(self) -> str:
        """
        :return: the mask type used
        """
        return self._mask_type

    @ModifierProp(serializable=True)
    def forward_sparsity(self) -> float:
        """
        :return: The sparsity enforced during inference.
        """
        return self._forward_sparsity

    @ModifierProp(serializable=True)
    def backward_sparsity(self) -> float:
        """
        :return: The sparsity enforced on the gradients during the backward pass.
        """
        return self._backward_sparsity

    @ModifierProp(serializable=False)
    def applied_sparsity(self) -> float:
        """
        :return: The applied forward sparsity.
        """
        return self._applied_sparsity

    @ModifierProp(serializable=True)
    def active_weight_decay(self) -> float:
        """
        :return: The magnitude of the weight decay on active parameters.
        """
        return self._active_weight_decay

    def _get_scorer(self, params: List[Parameter]) -> PruningParamsScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
        :note that both the weight and gradient masks are scored
        :by magnitude, so even though we have two sets of masks, only one
        :scorer is needed.  # TODO or is it?
        """
        return MagnitudePruningParamsScorer(params)

    def _get_mask_creator(
        self, param_names: List[str], params: List[Parameter]
    ) -> PruningMaskCreator:
        """
        :param names: full names of parameters to be pruned
        :param params: list of Parameters to be masked
        :return: mask creator object to be used by this pruning algorithm
        """
        return get_mask_creator_default(self.mask_type)

    def check_mask_update(
        self,
        module: Module,
        epoch: float,
        steps_per_epoch: int,
        recomputation_sparsity: Optional[float] = None,
        **kwargs,
    ):
        """
        Update mask values if necessary

        :param module: module to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        :param recomputation_sparsity: target sparsity for masks
        """
        started = self.started
        if self.start_pending(epoch, steps_per_epoch):
            self._module_masks.enabled = True
            self._grad_module_masks.enabled = True
            started = True

        if not self._pre_step_completed:
            # do pre optim step before mask update on update steps
            self._module_masks.pre_optim_step_update()
            self._grad_module_masks.pre_optim_step_update()
            self._pre_step_completed = True

        if started:
            # get sparsity level to be applied
            self._applied_sparsity = self.get_applied_sparsity_for_epoch(
                epoch, steps_per_epoch
            )
            self._grad_applied_sparsity = self.get_applied_grad_sparsity_for_epoch(
                epoch, steps_per_epoch
            )

            self._module_masks.update_param_masks(
                target=recomputation_sparsity or self._applied_sparsity
            )
            self._grad_module_masks.update_param_masks(
                target=recomputation_sparsity or self._grad_applied_sparsity
            )
            self._sparsity_applied = True

        if self.end_pending(epoch, steps_per_epoch):
            self._module_masks.pruning_end(leave_enabled=self._leave_enabled)
            self._grad_module_masks.pruning_end(leave_enabled=False)

    def finalize(self, module: Optional[Module] = None, reset_loggers: bool = True):
        """
        Cleans up any remaining hooks

        :param module: The model/module to finalize the modifier for.
            Marked optional so state can still be cleaned up on delete,
            but generally should always be passed in.
        :param reset_loggers: True to remove any currently attached loggers (default),
            False to keep the loggers attached.
        """
        super().finalize(module, reset_loggers)
        self._grad_module_masks.enabled = False
        self._grad_module_masks = None

    def state_dict(self) -> Dict[str, Tensor]:
        """
        :return: PyTorch state dictionary to store any variables from this modifier.
            The mapping is param_name -> mask
        """
        return OrderedDict(
            {
                "param_masks": OrderedDict(
                    zip(self._module_masks.names, self._module_masks.param_masks)
                ),
                "grad_masks": OrderedDict(
                    zip(
                        self._grad_module_masks.names,
                        self._grad_module_masks.param_masks,
                    )
                ),
            }
        )

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True):
        """
        Loads the given state dict into this object's modifiers

        :param state_dict: dictionary object as generated by this object's state_dict
            function
        :param strict: Ignored for this modifier, everything is treated as strict
        :raises IndexError: If any keys in the state dict do not correspond to a valid
            index for this manager and strict=True
        """
        if not self.initialized:
            raise RuntimeError("Cannot load state dict for an uninitialized modifier")

        mask_names = {key for key in self._module_masks.names}
        grad_mask_names = {key for key in self._grad_module_masks.names}
        state_dict_mask_keys = {key for key in state_dict["param_masks"].keys()}
        diff = mask_names.symmetric_difference(state_dict_mask_keys)
        if diff and strict:
            raise IndexError(
                f"Found extra keys: {state_dict_mask_keys - mask_names} "
                f"and missing keys: {mask_names - state_dict_mask_keys}"
            )
        state_dict_grad_mask_keys = {key for key in state_dict["grad_masks"].keys()}
        diff = grad_mask_names.symmetric_difference(state_dict_grad_mask_keys)
        if diff and strict:
            raise IndexError(
                f"Found extra keys: {state_dict_grad_mask_keys - grad_mask_names} "
                f"and missing keys: {grad_mask_names - state_dict_grad_mask_keys}"
            )

        self._module_masks.set_param_masks(
            [state_dict["param_masks"][name] for name in self._module_masks.names]
        )
        self._grad_module_masks.set_param_masks(
            [state_dict["grad_masks"][name] for name in self._grad_module_masks.names]
        )

    def _create_pruning_mask(
        self, layers: List[Module], layer_names: List[str], param_names: List[str]
    ) -> ModuleParamPruningMask:
        return ModuleParamPruningMask(
            layers,
            mask_creator=self._mask_creator,
            scorer=self._scorer,
            param_names=param_names,
            layer_names=layer_names,
            global_sparsity=self._global_sparsity,
            allow_reintroduction=True,
        )

    def _create_grad_pruning_mask(
        self, layers: List[Module], layer_names: List[str], param_names: List[str]
    ) -> ModuleParamPruningMask:
        return ModuleParamPruningMask(
            layers,
            mask_creator=self._grad_mask_creator,
            scorer=self._grad_scorer,
            param_names=param_names,
            layer_names=layer_names,
            global_sparsity=self._global_sparsity,
            mask_gradients_only=True,
        )

    @BasePruningModifier.log_call
    def optimizer_pre_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Performs any tracking or updates before the optimizer step is applied. Useful
        for tracking gradient values between backwards pass and optimizer step if
        optimizer clips gradients
        Weights are decayed according to the following formula:
        For weights active in the forward pass:
          w_i = w_i - w_i * self._active_weight_decay * current learning rate
        For weights inactive in the forward pass but active in
        the backward pass:
          w_i = w_i - w_i * 1/forward_sparsity * self._active_weight_decay * \
                  current learning rate

        The reason that we multiply by the learning rate is that in the original
        paper, this weight decay is presented as an L2 regularization
        term in the loss, and so is subject to the learning rate. It also
        keeps the magnitude of this term even with the impact of learing.
        Note that unlike computing L2 as a loss term, this implementation
        does not factor into the optimizer momentum. This is a deliberate
        choice, both because it seems to make sense in the context of
        Top-KAST, and because this is far simpler and cleaner to implement.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().optimizer_pre_step(module, optimizer, epoch, steps_per_epoch)

        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        with torch.no_grad():
            for i, param in enumerate(self._module_masks._params):
                param -= (
                    self._active_weight_decay
                    * lr
                    * self._module_masks._param_masks[i]
                    * param
                )

                param -= (
                    self._active_weight_decay
                    * lr
                    * (1 / self.forward_sparsity)
                    * (1 - self._module_masks.param_masks[i])
                    * self._grad_module_masks.param_masks[i]
                    * param
                )
