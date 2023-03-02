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
import math
import torch
import logging
import numpy as np

from torch import Tensor
from torch.nn import Module, Parameter
from typing import List, Optional, Union

from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.mask_creator import (
    PruningMaskCreator,
    get_mask_creator_default,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_base import (
    BaseGradualPruningModifier,
)
from sparseml.pytorch.utils import GradSampler
from sparseml.pytorch.utils.logger import BaseLogger
from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsGradScorer


__all__ = ["RigLPruningModifier", "RigLPruningParamsScorer"]


_LOGGER = logging.getLogger(__name__)


def cosine_schedule(t: float, t_max: float, init_value: float, end_value: float):
    """
    Cosine interpolation from init_value to end_value given by the law:
    f(t) = end_value + (1/2) * (init_value - end_value) (1 + cos(\pi t / t_max))

    :param t: current timestep
    :param t_max: maximal timestep
    :param init_value: initial value
    :param end_value: final value at t_max
    """
    return end_value + (init_value - end_value) * 0.5 * (
        1 + math.cos(math.pi * t / t_max)
    )


def threshold_fraction(tensor: Tensor, fraction: float) -> None:
    """
    A function returning the tensor with all but topk fraction
    elements set to 0.

    :param tensor: the input tensor
    :param fraction: fraction of nonzero elements
    """
    lookup_idx = round(fraction * tensor.numel())
    if lookup_idx == 0:
        return tensor
    threshold, _ = torch.kthvalue(tensor.reshape(-1), k=lookup_idx)
    return torch.where(tensor > threshold, tensor, 0.0)


@PyTorchModifierYAML()
class RigLPruningModifier(BaseGradualPruningModifier):
    """
    As described in https://arxiv.org/abs/1911.11134

    Sparse training procedure that starts from a sparse model
    with (1 - init_update_fraction) * sparsity and gradually increases
    the sparsity with a cosine schedule up to sparsity.
    At each update a fraction of weights
    init_update_fraction * cos(\pi (epoch - start_epoch) / (end_epoch - start_epoch))
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
    |       sparsity: 0.7
    |       start_epoch: 2.0
    |       end_epoch: 26.0
    |       update_frequency: 4.0
    |       params: ["re:.*weight"]
    |       leave_enabled: True
    |       inter_func: cubic
    |       global_sparsity: True
    |       mask_type: unstructured
    |       sparsity_strategy: erdos_renyi_kernel
    |       init_update_fraction: 0.3

    :param sparsity: the final sparsity for the param to end with at end_epoch.
        A single float for the whole model.
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to update at
        between start and end
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights. If a sparsity to param mapping is defined by
        final_sparsity, then params should be set to []
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: the type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param mask_type: String to define type of sparsity to apply.
        RigL modifier supports only 'unstructured'
    :param global_sparsity: set True to enable global pruning. If False, pruning will
        be layer-wise. Default is True
    :param sparsity_strategy: String to define the sparsity distribution. Following
        the original paper one can select one of the 3 options:
        [uniform, erdos_renyi, erdos_renyi_kernel].
    :param init_update_fraction: The initial percentage of the weights updated -
        pruned and regrown
    """

    _supported_masks = "unstructured"
    _sparsity_strategies = ("uniform", "erdos_renyi", "erdos_renyi_kernel")

    def __init__(
        self,
        sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        leave_enabled: bool = True,
        global_sparsity: bool = True,
        mask_type: str = "unstructured",
        sparsity_strategy: str = "erdos_renyi_kernel",
        init_update_fraction: float = 0.3,
    ):
        super().__init__(
            params=params,
            init_sparsity=sparsity * (1 - init_update_fraction),
            final_sparsity=sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            global_sparsity=global_sparsity,
            leave_enabled=leave_enabled,
            parent_class_kwarg_names=[],
        )
        self._mask_type = mask_type
        self._sparsity = sparsity
        self._sparsity_strategy = sparsity_strategy
        self._init_update_fraction = init_update_fraction
        # check arguments
        self._validate()

    def _validate(self):
        assert (
            self._mask_type in self._supported_masks
        ), f"{self._mask_type} mask_type not supported"

        if self._global_sparsity:
            assert self._sparsity_strategy in (
                "erdos_renyi",
                "erdos_renyi_kernel",
            ), "Global sparsity supports only `erdos_renyi`, `erdos_renyi_kernel`"
        else:
            assert (
                self._sparsity_strategy is "uniform"
            ), "Uniform sparsity support only `uniform`"

    @ModifierProp()
    def mask_type(self) -> str:
        """
        :return: the mask type used
        """
        return self._mask_type

    @ModifierProp(serializable=True)
    def sparsity(self) -> float:
        """
        :return: The initial sparsity for the variable to start with at start_epoch
        """
        return self._sparsity

    @ModifierProp(serializable=True)
    def sparsity_strategy(self) -> str:
        """
        :return: the mask type used
        """
        return self._sparsity_strategy

    @ModifierProp(serializable=True)
    def init_update_fraction(self) -> float:
        """
        :return: the mask type used
        """
        return self._init_update_fraction

    @ModifierProp(serializable=False)
    def init_sparsity(self) -> float:
        """
        :return: The initial sparsity for the variable to start with at start_epoch
        """
        return self._init_sparsity

    @ModifierProp(serializable=False)
    def final_sparsity(self) -> float:
        """
        :return: The initial sparsity for the variable to start with at start_epoch
        """
        return self._init_sparsity

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
        :return: sparsity level that should be applied based on the given interpolation
            function
        """
        return [
            cosine_schedule(
                epoch - self._start_epoch,
                self._end_epoch - self._start_epoch,
                self._final_sparsity * (1 - self._init_update_fraction),
                self._final_sparsity,
            )
            for _ in range(len(self.module_masks.layers))
        ]

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        **kwargs,
    ):
        """
        Grab the layers and apply if epoch in range to control pruning for.
        Expects `grad_sampler` dict with `data_loader_builder` and `loss_fn`
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
            and "loss_fn" not in kwargs["grad_sampler"]
        ):
            raise RuntimeError(
                "grad_sampler dict with data_loader_builder and loss_fn "
                "must be provided to initialize GradSampler"
            )

        self._grad_sampler = GradSampler(
            kwargs["grad_sampler"]["data_loader_builder"](),
            kwargs["grad_sampler"]["loss_fn"],
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

    def _get_update_fraction(self, epoch):
        """
        Returns the fraction of params updated at the current epoch.

        :param epoch: current epoch
        """
        return cosine_schedule(
            epoch - self._start_epoch,
            self._end_epoch - self._start_epoch,
            self._final_sparsity * self._init_update_fraction,
            0.0,
        )

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

        self.scorer._update_fraction = self._get_update_fraction(epoch)
        super().check_mask_update(module, epoch, steps_per_epoch, **kwargs)

    def _collect_grad_samples(
        self,
        module: Module,
        grad_sampler: GradSampler,
    ):
        if not isinstance(grad_sampler, GradSampler):
            raise ValueError(
                "One-shot OBS pruning requires an initialized GradSampler object."
                f"`The given object is of type {type(grad_sampler)}"
            )

        is_training = module.training
        _LOGGER.debug("Setting the model in the eval mode")
        module.eval()

        _LOGGER.debug(f"Collecting grad with GradSampler")
        # only a single update is needed
        for _ in grad_sampler.iter_module_backwards(module, 1):
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

    def sparsity_scaler(self, score: Tensor) -> float:
        """
        Assigns the sparsity scale for a given parameter
        according to the sparsity_strategy. The weights
        with smaller scale are pruned more than those with larger.
        """
        assert len(score.shape) >= 2, "Pruned weight must be at least 2-dimensional."
        if self._sparsity_strategy == "uniform":
            return 1.0
        elif self._sparsity_strategy == "erdos_renyi":
            c_out, c_in = score.shape[:2]
            return (c_in + c_out) / (c_in * c_out)
        elif self._sparsity_strategy == "erdos_renyi_kernel":
            return np.sum(score.shape) / np.prod(score.shape)
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
            cumulative_sum += self.sparsity_scaler(param) * param.numel()
            total_params += param.numel()
        norm_factor = ((1 - self._sparsity) * total_params) / cumulative_sum
        return [
            np.clip(1 - norm_factor * self.sparsity_scaler(param), 0.0, 1.0)
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

        for i, _ in enumerate(self._param_grads):
            self._param_grads[i] = self._params[i].grad.clone()

    def get_param_score(self, param: Tensor, param_grad: Tensor, param_sparsity: float):
        """
        Computed the saliency score for a given parameter.

        :param param: param to be scored
        :param param_grad: gradient of the parameters
        :param param_sparsity: the sparsity for a given parameter
        """
        magn_score = param.abs()
        magn_score = threshold_fraction(magn_score, param_sparsity)
        grad_score = param_grad.abs() * param.eq(0)
        grad_score = threshold_fraction(
            grad_score, 1 - self._update_fraction * param_sparsity
        )
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

        self._broadcast_list_from_main(scores)
        return scores
