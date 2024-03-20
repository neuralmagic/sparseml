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

"""
Modifier classes implementing the RigL sparse training procedure.
The algorithm is described in details in the
Rigging the Lottery: Making All Tickets Winners paper https://arxiv.org/abs/1911.11134.
"""
import logging
import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.mask_creator import (
    PruningMaskCreator,
    get_mask_creator_default,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_base import (
    BaseGradualPruningModifier,
)
from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsGradScorer
from sparseml.pytorch.utils import GradSampler
from sparseml.pytorch.utils.logger import BaseLogger
from sparseml.utils import FROM_PARAM_TOKEN


__all__ = ["RigLPruningModifier", "RigLPruningParamsScorer"]


_LOGGER = logging.getLogger(__name__)


def cosine_schedule(t: float, t_max: float, init_value: float, end_value: float):
    """
    Cosine interpolation from init_value to end_value given by the law:
    f(t) = end_value + (1/2) * (init_value - end_value) (1 + cos(pi t / t_max))

    :param t: current timestep
    :param t_max: maximal timestep
    :param init_value: initial value
    :param end_value: final value at t_max
    """
    return end_value + (init_value - end_value) * 0.5 * (
        1 + math.cos(math.pi * t / t_max)
    )


def threshold_fraction(
    tensor: Tensor, fraction: float, set_to_max: bool = False
) -> None:
    """
    A function returning the tensor with all but topk fraction
    elements set to 0.

    :param tensor: the input tensor
    :param fraction: fraction of zero elements
    """
    lookup_idx = round((1 - fraction) * tensor.numel())
    if lookup_idx == 0:
        return torch.zeros_like(tensor)
    tensor_shape = tensor.shape
    vals, idx = tensor.reshape(-1).topk(lookup_idx, largest=True)
    topk = torch.zeros_like(tensor.reshape(-1))
    if set_to_max:
        topk[idx] = torch.finfo(tensor.dtype).max
    else:
        topk[idx] = vals
    return topk.reshape(tensor_shape)


@PyTorchModifierYAML()
class RigLPruningModifier(BaseGradualPruningModifier):
    """
    As described in https://arxiv.org/abs/1911.11134

    Sparse training procedure that trains a model with (1-final sparsity) parameters.
    At each update a fraction of weights
    init_update_fraction * cos(pi (epoch - start_epoch) / (end_epoch - start_epoch))
    with smallest magnitude is pruned and the same
    amount of weights are regrown according to
    the magnitude of the gradient.

    Pruner supports three sparsity strategies as in the original paper:
    - uniform
    - erdos_renyi
    - erdos_renyi_kernel

    Supported mask types: unstructured.

    | Sample yaml:
    |   !RigLPruningModifier
    |       final_sparsity: 0.7
    |       start_epoch: 2.0
    |       end_epoch: 26.0
    |       update_frequency: 4.0
    |       num_grads: 100
    |       params: ["re:.*weight"]
    |       global_sparsity: False
    |       leave_enabled: True
    |       mask_type: unstructured
    |       sparsity_strategy: "erdos_renyi_kernel"
    |       init_update_fraction: 0.3
    |       grad_sampler_kwargs:
    |           batch_size: 256

    :param final_sparsity: the final sparsity for the param to end with at end_epoch.
        A single float for all prunable layers of the whole model. Note that
        init_sparsity is always automatically set to final_sparsity.
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to update at
        between start and end
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights. If a sparsity to param mapping is defined by
        final_sparsity, then params should be set to []
    :param global_sparsity: set True to enable global pruning. If False, pruning will
        be layer-wise. Must be set to False, as global sparsity is not supported yet.
    :param momentum_buffer_reset: set True to reset momentum buffer
        for pruned weights at every optimizer step, so that reintroduced
        weights start with an empty momentum buffer.
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param mask_type: String to define type of sparsity to apply.
        RigL modifier supports only 'unstructured'
    :param num_grads: Number of grads to be collected by the grad sampler for
        recomputing the mask.
    :param sparsity_strategy: String to define the sparsity distribution. Following
        the original paper one can select one of the 3 options:
        [uniform, erdos_renyi, erdos_renyi_kernel].
    :param init_update_fraction: The initial percentage of the weights updated -
        pruned and regrown
    :param grad_sampler_kwargs: kwargs to override default train dataloader config
            for pruner's gradient sampling.
    """

    _supported_masks = ["unstructured"]
    _sparsity_strategies = ("uniform", "erdos_renyi", "erdos_renyi_kernel")

    def __init__(
        self,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        num_grads: int = 1,
        leave_enabled: bool = True,
        global_sparsity: bool = False,
        momentum_buffer_reset: bool = True,
        mask_type: str = "unstructured",
        sparsity_strategy: str = "erdos_renyi_kernel",
        init_update_fraction: float = 0.3,
        grad_sampler_kwargs: Optional[Dict[str, Any]] = {},
        **kwargs,
    ):
        self._sparsity_strategy = sparsity_strategy
        super().__init__(
            params=params,
            final_sparsity=final_sparsity,
            init_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            global_sparsity=False,
            update_frequency=update_frequency,
            leave_enabled=leave_enabled,
            parent_class_kwarg_names=[],
            **kwargs,
        )
        # self._sparsity_distribution = self._scorer.get_sparsity_distribution()
        self._mask_type = mask_type
        # self._sparsity_strategy = sparsity_strategy
        self._momentum_buffer_reset = momentum_buffer_reset
        self._init_update_fraction = init_update_fraction
        self._grad_sampler_kwargs = grad_sampler_kwargs
        self._num_grads = num_grads
        self._validate()

    def _validate(self):
        # The momentum buffer reset code depends on parameters being explicitly
        # set to 0 by the pruning_masks, which does not happen if _allow_reintroduction
        # is set to True. This is an artifact of the implementation, see TODO below.
        assert not (
            self._allow_reintroduction and self._momentum_buffer_reset
        ), "Allow_reintroduction and momentum_buffer_reset cannot both be true."

        if self._final_sparsity == FROM_PARAM_TOKEN:
            raise ValueError(f"{FROM_PARAM_TOKEN} is not supported for RigL Pruning.")

        assert (
            self._mask_type in self._supported_masks
        ), f"{self._mask_type} mask_type not supported"

        if self._global_sparsity:
            raise ValueError("global sparsity is not supported for RigL.")
        else:
            assert self._sparsity_strategy in (
                "erdos_renyi",
                "erdos_renyi_kernel",
                "uniform",
            ), "erdos_renyi, erdos_renyi_kernel, and uniform sparsity are supported."

    # Override te optimizer_post_step method to reset the momentum of pruned weights.
    # TODO: this implementation has some  dependencies that may be better handled
    # a different way in the future:
    # First:
    # we rely on the BasePruningModifier method to apply the mask to the weights in its
    # own optimizer_post_step method. If this logic is moved or skipped (as it is today
    # when allow_reintroduction is true), then the trick of zeroing out momentum where
    # the weights are 0 won't work.
    # Second:
    # we explicitly zero out the optimizer momentum-related buffers here, which also
    # zeros out some very specific ones, like the ones for Adam. If future optimizers
    # have other ways of storing this information, then this code won't know about it/
    # zero them out.
    # Unfortunately this may not be easy to fix without a larger code restructuring;
    # Fortunately, this will likely not become an issue anytime soon, and even if
    # something does change, the impact on model quality is not likely to be at all
    # significant. Therefore, left as TODO.
    def optimizer_pre_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Reset momentum buffer if mask was just updated.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """

        super().optimizer_pre_step(module, optimizer, epoch, steps_per_epoch)

        if self._momentum_buffer_reset:
            """
            This condition is only evaluated when `momentum_buffer_reset`
            strategy is True. We set the momentum of all masked weights to zero,
            so that if the weights get reintroduced, they truly start from 0.
            """
            self._reset_momentum_buffer(optimizer)

    @ModifierProp()
    def mask_type(self) -> str:
        """
        :return: the mask type used
        """
        return self._mask_type

    @ModifierProp(serializable=True)
    def global_sparsity(self) -> str:
        """
        :return: the mask type used
        """
        return self._global_sparsity

    @ModifierProp(serializable=True)
    def init_update_fraction(self) -> float:
        """
        :return: the initial maks update fraction
        """
        return self._init_update_fraction

    @ModifierProp(serializable=False)
    def init_sparsity(self) -> float:
        """
        :return: The initial sparsity for the variable to start with at start_epoch
        """
        return self._final_sparsity

    @ModifierProp(serializable=True)
    def final_sparsity(self) -> float:
        """
        :return: the final sparsity for the variable to start with at end_epoch
        """
        return self._final_sparsity

    @ModifierProp(serializable=True)
    def sparsity_strategy(self) -> str:
        """
        :return: the sparsification strategy for the pruner (uniform, ER, ERK)
        """
        return self._sparsity_strategy

    @ModifierProp(serializable=True)
    def momentum_buffer_reset(self) -> bool:
        """
        :return: whether the momentum buffer should be reset for pruned weights
        """
        return self._momentum_buffer_reset

    @ModifierProp(serializable=True)
    def num_grads(self) -> int:
        """
        :return: The number of gradient batches to collect for sampling
        """
        return self._num_grads

    @ModifierProp()
    def grad_sampler_kwargs(self) -> Optional[Dict[str, Any]]:
        """
        :return: dict of training dataloader's overridden configs for gradient sampling
        """
        return self._grad_sampler_kwargs

    def _get_mask_creator(
        self, param_names: List[str], params: List[Parameter]
    ) -> PruningMaskCreator:
        """
        :param names: full names of parameters to be pruned
        :param params: list of Parameters to be masked
        :return: mask creator object to be used by this pruning algorithm
        """
        return get_mask_creator_default(self.mask_type)

    def get_applied_sparsity_for_epoch(
        self, epoch: float, steps_per_epoch: int
    ) -> Union[float, List[float]]:
        """
        :param epoch: current epoch
        :param steps_per_epoch: number of steps in each epoch
        :return: sparsity level that should be applied (always final_sparsity)
        """

        self._sparsity_distribution = self._scorer.get_sparsity_distribution()
        _LOGGER.info(f"RigL applied sparsity {self._sparsity_distribution}")
        return self._sparsity_distribution

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        **kwargs,
    ):
        """
        Grab the layers and apply if epoch in range to control pruning for.
        Expects `grad_sampler` dict with `data_loader_builder` and `loss_function`
        to initialize GradSampler instance and optionally override data-loader's
        hyperparams with `grad_sampler` given in the recipe.

        :param module: the PyTorch model/module to modify
        :param epoch: the epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: optional list of loggers to log the modification process to
        :param kwargs: optional kwargs to support specific arguments
            for individual modifiers.
        """
        if (
            "data_loader_builder" not in kwargs["grad_sampler"]
            and "loss_function" not in kwargs["grad_sampler"]
        ):
            raise RuntimeError(
                "grad_sampler dict with data_loader_builder and loss_function "
                "must be provided to initialize GradSampler"
            )

        self._grad_sampler = GradSampler(
            kwargs["grad_sampler"]["data_loader_builder"](**self._grad_sampler_kwargs),
            kwargs["grad_sampler"]["loss_function"],
        )

        super().initialize(module, epoch, loggers, **kwargs)

    def _get_scorer(self, params: List[Parameter]) -> PruningParamsGradScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
        """
        return RigLPruningParamsScorer(
            params=params,
            sparsity=self._final_sparsity,
            sparsity_strategy=self._sparsity_strategy,
        )

    @staticmethod
    def _reset_momentum_buffer(optimizer):
        # multiply momentum buffer my param mask
        if "state" in optimizer.state_dict():
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    state = optimizer.state[param]
                    param_mask = param.ne(0)
                    if "momentum_buffer" in state:
                        state["momentum_buffer"].mul_(param_mask)
                    # Adam-specific additional buffers.
                    if "exp_avg" in state:
                        state["exp_avg"].mul_(param_mask)
                    if "exp_avg_sq" in state:
                        state["exp_avg"].mul_(param_mask)
                    if "max_exp_avg_sq" in state:
                        state["exp_avg"].mul_(param_mask)

    def _get_update_fraction(self, epoch):
        """
        Returns the fraction of params updated at the current epoch.

        :param epoch: current epoch
        """
        update_fraction = cosine_schedule(
            epoch - self._start_epoch,
            self._end_epoch - self._start_epoch,
            self._init_update_fraction,
            0.0,
        )
        _LOGGER.info(f"RigL mask Update fraction {update_fraction}")
        return update_fraction

    def check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        # enable grad buffering
        torch.cuda.empty_cache()
        if self._scorer._is_main_proc:
            self._scorer._enabled_grad_buffering = True
        # reset masks
        for mask in self._module_masks._param_masks:
            mask.data = torch.ones_like(mask)
        # collect grad
        self._collect_grad_samples(module, self._grad_sampler)
        self._pre_step_completed = True
        # disable grad buffering
        torch.cuda.empty_cache()
        if self._scorer._is_main_proc:
            self._scorer._enabled_grad_buffering = False

        self.scorer.set_update_fraction(self._get_update_fraction(epoch))
        super().check_mask_update(module, epoch, steps_per_epoch, **kwargs)

    def _collect_grad_samples(
        self,
        module: Module,
        grad_sampler: GradSampler,
    ):
        if not isinstance(grad_sampler, GradSampler):
            raise ValueError(
                "RigL pruning requires an initialized GradSampler object."
                f"`The given object is of type {type(grad_sampler)}"
            )

        is_training = module.training
        _LOGGER.debug("Setting the model in the eval mode")
        module.eval()

        _LOGGER.debug("Collecting grad with GradSampler")
        for _ in grad_sampler.iter_module_backwards(module, self._num_grads):
            self._module_masks.pre_optim_step_update()

        if is_training:
            _LOGGER.debug("Setting the model back to the train mode")
            module.train()


class RigLPruningParamsScorer(PruningParamsGradScorer):
    """
    Scores parameters using the criteria introduced in the
    Rigging the Lottery: Making All Tickets Winners paper.
    At each update iteration a certain fraction of weights
    with smallest magnitude is pruned and the same fraction
    is regrown (i.e sparsity mask for these parameters is removed)
    according to the magnitude of the gradient criterion.

    :param params: list of model Parameters to track and score
    :param sparsity: the final model sparsity
    :param sparsity_strategy:
    """

    def __init__(
        self,
        params: List[Parameter],
        sparsity: float,
        sparsity_strategy: str = "erdos_renyi_kernel",
    ):
        super().__init__(params)
        self._params = params
        self._sparsity = sparsity
        self._sparsity_strategy = sparsity_strategy
        self._update_fraction = None
        # compute sparsity distribution
        self._sparsity_distribution = self.get_sparsity_distribution()
        # prepare grad storage
        self._param_grads = [None] * len(params)
        self._enabled_grad_buffering = False

    def set_update_fraction(self, value: float):
        """
        :params value: the new value of _update_fraction
        """
        self._update_fraction = value

    def density_scaler(self, param: Tensor) -> float:
        """
        Assigns the density weights for a given parameter
        according to the sparsity_strategy. The layers
        with larger weights are pruned less (the density
        will be proportional to the weight).
        """
        assert len(param.shape) >= 2, "Pruned weight must be at least 2-dimensional."
        if self._sparsity_strategy == "uniform":
            return 1.0
        elif self._sparsity_strategy == "erdos_renyi":
            c_out, c_in = param.shape[:2]
            return (c_in + c_out) / (c_in * c_out)
        elif self._sparsity_strategy == "erdos_renyi_kernel":
            return np.sum(param.shape) / np.prod(param.shape)
        else:
            raise ValueError("Unknown sparsity distribution")

    @torch.no_grad()
    def get_sparsity_distribution(self) -> None:
        """
        Returns the sparsity distribution for every parameter
        scaled with respect to the sparsity strategy.
        """
        total_params = 0
        cumulative_sum = 0
        for param in self._params:
            cumulative_sum += self.density_scaler(param) * param.numel()
            total_params += param.numel()
        norm_factor = ((1 - self._sparsity) * total_params) / cumulative_sum
        return [
            np.clip(1 - norm_factor * self.density_scaler(param), 0.0, 1.0)
            for param in self._params
        ]

    @torch.no_grad()
    def pre_optim_step_update(self, masks: List[Tensor]):
        """
        Collect single gradient for estimation of RigL saliency score.

        :param masks: latest masks that are applied to these parameters
        """
        if not self._enabled_grad_buffering:
            # only collect gradients when called during pruning step
            # this ignores calls invoked by manager during training
            return

        for i, param_grad in enumerate(self._param_grads):
            if param_grad is None:
                self._param_grads[i] = self._params[i].grad.clone()
            else:
                self._param_grads[i].add_(self._params[i].grad)

    def get_param_score(self, param: Tensor, param_grad: Tensor, param_sparsity: float):
        """
        Computed the saliency score for a given parameter.

        :param param: param to be scored
        :param param_grad: gradient of the parameters
        :param param_sparsity: the sparsity for a given parameter
        """
        # We drop and replace mask_update_fraction of nonzero elements
        mask_update_fraction = (1 - param_sparsity) * self._update_fraction
        magn_score = param.abs()
        # Of the existing mask, we keep the top 1-param_sparsity-mask_update_fraction
        # elements by magnitude.
        magn_score = threshold_fraction(
            magn_score, (param_sparsity + mask_update_fraction), set_to_max=True
        )
        # For the rest of the unmasked weights, we use the gradient magnitude
        # as the criterion. These cannot be larger than the magnitude scores,
        # since those are set to the largest possible value.
        grad_score = (param_grad.abs()) * magn_score.eq(0)
        score = magn_score + grad_score
        return score

    def score_parameters(self) -> List[Tensor]:
        """
        :return: List of Tensors the same shapes as the given Parameters where
            each Parameter's elements are scored according to criterion
            introduced in Rigging the Lottery: Making All Tickets Winners paper.
        """
        scores = [None] * len(self._params)
        if self._is_main_proc:
            scores = [
                self.get_param_score(param, param_grad, param_sparsity)
                for param, param_grad, param_sparsity in zip(
                    self._params, self._param_grads, self._sparsity_distribution
                )
            ]

        # free memory
        for i, _ in enumerate(self._param_grads):
            self._param_grads[i] = None
        self._broadcast_list_from_main(scores)
        return scores
