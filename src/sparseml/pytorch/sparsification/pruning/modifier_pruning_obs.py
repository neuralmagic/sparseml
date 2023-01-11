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
Modifier classes implementing the blockwise version of the Optimal Brain Surgeon
pruning framework, optimized for small blocks. The algorithm is described in details
in the Optimal BERT Surgeon paper https://arxiv.org/abs/2203.07259
"""
import logging
import math
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter

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
from sparseml.utils import interpolate


__all__ = [
    "OBSPruningModifier",
    "OBSPruningParamsScorer",
    "EmpiricalBlockFisherInverse",
]


_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class OBSPruningModifier(BaseGradualPruningModifier):
    """
    As described in https://arxiv.org/abs/2203.07259

    Gradually applies sparsity to a given parameter or parameters from
    init_sparsity until final_sparsity is reached over a given number of epochs.
    Uses the Optimal BERT Surgeon algorithm to prune weights based on the
    approximate second-order information of the loss function. When pruning,
    it also updates remaining weights to compensate for accuracy drops incurred
    by pruning. It follows the Optimal Brain Surgeon framework with approximations
    and optimizations to make it efficient but accurate for huge models.
    It can be used to prune other models besides BERT too.

    Naming convention with respect to the paper:
        * damp == small dampening constant 'lambda'
        * num_grads == number of gradient outer products 'm'
        * fisher_block_size == size of the blocks 'B' along the main diagonal

    Memory requirements: O(dB), where 'd' is the total number of prunable weights.
    If O(dB) can't fit on a single GPU device, pytorch DDP should be used to split
    the computational overhead equally between devices.

    Supported mask types: unstructured and block4.

    | Sample yaml:
    |   !OBSPruningModifier
    |       init_sparsity: 0.7
    |       final_sparsity: 0.9
    |       start_epoch: 2.0
    |       end_epoch: 26.0
    |       update_frequency: 4.0
    |       params: ["re:.*weight"]
    |       leave_enabled: True
    |       inter_func: cubic
    |       global_sparsity: True
    |       mask_type: unstructured
    |       num_grads: 1024
    |       damp: 1e-7
    |       fisher_block_size: 50
    |       grad_sampler_kwargs:
    |           batch_size: 8

    :param init_sparsity: the initial sparsity for the param to start with at
        start_epoch
    :param final_sparsity: the final sparsity for the param to end with at end_epoch.
        Can also be a Dict of final sparsity values to a list of parameters to apply
        them to. If given a Dict, then params must be set to [] and the params to
        be pruned will be read from the final_sparsity Dict
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
    :param mask_type: String to define type of sparsity to apply. 'unstructured'
        and 'block4' are supported. Default is 'unstructured'
    :param global_sparsity: set True to enable global pruning. If False, pruning will
        be layer-wise. Default is True
    :param num_grads: number of gradients used to calculate the Fisher approximation
    :param damp: dampening factor, default is 1e-7
    :param fisher_block_size: size of blocks along the main diagonal of the Fisher
        approximation, default is 50
    :param grad_sampler_kwargs: kwargs to override default train dataloader config
        for pruner's gradient sampling.
    :param num_recomputations: number of recomputations of the inverse Hessian
        approximation while performing one pruning step
    """

    def __init__(
        self,
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        global_sparsity: bool = True,
        mask_type: str = "unstructured",
        num_grads: int = 1024,
        damp: float = 1e-7,
        fisher_block_size: int = 50,
        grad_sampler_kwargs: Optional[Dict[str, Any]] = None,
        num_recomputations: int = 1,
    ):
        super().__init__(
            params=params,
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            inter_func=inter_func,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            global_sparsity=global_sparsity,
            leave_enabled=leave_enabled,
            parent_class_kwarg_names=[],
        )
        self._mask_type = mask_type
        self._num_grads = num_grads
        self._damp = damp
        self._fisher_block_size = fisher_block_size
        self._grad_sampler_kwargs = grad_sampler_kwargs
        self._num_recomputations = num_recomputations
        self._last_applied_sparsity = 0.0  # keep track for recomputations

        self._grad_sampler = None
        self._supported_masks = ("unstructured", "block4")

        self._validate()

    @ModifierProp()
    def mask_type(self) -> str:
        """
        :return: the mask type used
        """
        return self._mask_type

    @ModifierProp()
    def num_grads(self) -> int:
        """
        :return: number of gradients used to calculate the Fisher approximation
        """
        return self._num_grads

    @ModifierProp()
    def damp(self) -> float:
        """
        :return: dampening factor used for inverse Fisher calculation
        """
        return self._damp

    @ModifierProp()
    def fisher_block_size(self) -> int:
        """
        :return: size of blocks along the main diagonal of the Fisher approximation
        """
        return self._fisher_block_size

    @ModifierProp()
    def grad_sampler_kwargs(self) -> Optional[Dict[str, Any]]:
        """
        :return: dict of training dataloader's overridden configs for gradient sampling
        """
        return self._grad_sampler_kwargs

    @ModifierProp()
    def num_recomputations(self) -> int:
        """
        :return: number of recomputations of the inverse Hessian approximation
            while doing one pruning step
        """
        return self._num_recomputations

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
        hyperparams with `grad_sampler_kwargs` given in the recipe.

        :param module: the PyTorch model/module to modify
        :param epoch: the epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: optional list of loggers to log the modification process to
        :param kwargs: optional kwargs to support specific arguments
            for individual modifiers.
        """
        if (
            "grad_sampler" not in kwargs
            or "data_loader_builder" not in kwargs["grad_sampler"]
            or "loss_function" not in kwargs["grad_sampler"]
        ):
            raise RuntimeError(
                "grad_sampler dict with data_loader_builder and loss_function "
                "must be provided to initialize GradSampler"
            )

        self._grad_sampler = GradSampler(
            kwargs["grad_sampler"]["data_loader_builder"](self._grad_sampler_kwargs),
            kwargs["grad_sampler"]["loss_function"],
        )

        super().initialize(module, epoch, loggers, **kwargs)

    def check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        if steps_per_epoch == 1 and not math.isinf(epoch):
            return  # not a one-shot run

        torch.cuda.empty_cache()
        if self._scorer._is_main_proc:
            _LOGGER.info("Running OBS Pruning")
            self._scorer._enabled_grad_buffering = True

        self._pre_step_completed = True
        to_apply_sparsities = self.get_applied_sparsity_for_epoch(
            epoch, steps_per_epoch
        )
        last_applied_sparsities = (
            self._last_applied_sparsity
            if isinstance(self._last_applied_sparsity, List)
            else [self._last_applied_sparsity] * len(to_apply_sparsities)
        )

        for i in range(1, self._num_recomputations + 1):
            self._collect_grad_samples(module, self._grad_sampler)
            recomputation_sparsity = [
                interpolate(
                    i,
                    0,
                    self._num_recomputations,
                    start_sparsity,
                    target_sparsity,
                )
                for start_sparsity, target_sparsity in zip(
                    last_applied_sparsities, to_apply_sparsities
                )
            ]
            super().check_mask_update(
                module,
                epoch,
                steps_per_epoch,
                recomputation_sparsity=recomputation_sparsity,
            )

        torch.cuda.empty_cache()
        self._last_applied_sparsity = to_apply_sparsities
        if self._scorer._is_main_proc:
            self._scorer._enabled_grad_buffering = False

    def _get_mask_creator(
        self, param_names: List[str], params: List[Parameter]
    ) -> PruningMaskCreator:
        """
        :param names: full names of parameters to be pruned
        :param params: list of Parameters to be masked
        :return: mask creator object to be used by this pruning algorithm
        """
        return get_mask_creator_default(self.mask_type)

    def _get_scorer(self, params: List[Parameter]) -> PruningParamsGradScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
        """
        return OBSPruningParamsScorer(
            params=params,
            num_grads=self._num_grads,
            damp=self._damp,
            fisher_block_size=self._fisher_block_size,
            mask_type=self._mask_type,
        )

    def _collect_grad_samples(
        self,
        module: Module,
        grad_sampler: GradSampler,
    ):
        if not isinstance(grad_sampler, GradSampler):
            raise ValueError(
                "One-shot OBS pruning requires a GradSampler object given by the "
                f"grad_sampler kwarg. Given an object of type {type(grad_sampler)}"
            )

        is_training = module.training
        _LOGGER.debug("Setting the model in the eval mode")
        module.eval()

        _LOGGER.debug(f"Starting to collect {self._num_grads} grads with GradSampler")
        for _ in grad_sampler.iter_module_backwards(module, self._num_grads):
            self._module_masks.pre_optim_step_update()

        if is_training:
            _LOGGER.debug("Setting the model back to the train mode")
            module.train()

    def _validate(self):
        if not hasattr(torch.linalg, "solve"):
            raise RuntimeError(
                f"torch>=1.9 required to use {self.__class__.__name__} "
                f"found {torch.__version__}"
            )

        if isinstance(self._damp, str):  # to support 'damp: 1e-7' in the recipe
            self._damp = float(self._damp)

        if self._mask_type not in self._supported_masks:
            raise ValueError(f"{self._mask_type} mask_type not supported")

        if self._mask_type == "block4" and self._fisher_block_size % 4 != 0:
            raise ValueError(
                "fisher_block_size must be divisible by 4 for block4 pruning"
            )


class OBSPruningParamsScorer(PruningParamsGradScorer):
    """
    Scores parameters using the equations introduced in the Optimal BERT Surgeon
    to solve for the optimal weight update in the Optimal Brain Surgeon (OBS)
    framework. Implements unstructured and semi-structured (block4) scoring and
    pruning.

    :param params: list of model Parameters to track and score
    :param num_grads: number of gradients used to calculate the Fisher approximation
    :param damp: dampening factor, default is 1e-7
    :param fisher_block_size: size of blocks along the main diagonal of the Fisher
        approximation, default is 50
    """

    def __init__(
        self,
        params: List[Parameter],
        num_grads: int,
        damp: float,
        fisher_block_size: int,
        mask_type: str,
    ):
        super().__init__(params)
        self._num_grads = num_grads
        self._damp = damp
        self._fisher_block_size = fisher_block_size
        self._mask_type = mask_type

        self._finvs = None  # type: List[EmpiricalBlockFisherInverse]
        self._enabled_grad_buffering = False
        self._eps = torch.finfo(torch.float32).eps

        # assign device to each Finv
        self._devices = []
        num_devices = torch.cuda.device_count()
        if num_devices == 0:
            self._devices = [torch.device("cpu")] * len(self._params)
        else:
            num_devices = min(num_devices, len(self._params))
            per_device = math.floor(len(self._params) / num_devices)
            for i in range(num_devices):
                self._devices += [torch.device("cuda", i)] * per_device
            remainder = len(self._params) - len(self._devices)
            if remainder > 0:
                self._devices += [self._devices[-1]] * remainder

        self._pickle_exclude_params.extend(
            [
                "_finvs",
                "_enabled_grad_buffering",
                "_devices",
            ]
        )
        self._validate()

    @torch.no_grad()
    def score_parameters(self) -> List[Tensor]:
        """
        :return: List of Tensors the same shapes as the given Parameters where
            each Parameter's elements are scored based on the blockwise OBS
        """
        scores = [None] * len(self._params)
        block_finv_w = [None] * len(self._params)

        if self._is_main_proc:
            for i, finv in enumerate(self._finvs):
                if self._mask_type == "unstructured":
                    scores[i] = (
                        (self._params[i].data.view(-1) ** 2).to(self._devices[i])
                        / (2.0 * finv.diag() + self._eps)
                    ).view(self._params[i].shape)
                else:  # self._mask_type == "block4":
                    block_w = self._params[i].data.view(-1, 4).to(finv.dev)  # (d/Q, Q)
                    block_finv = (
                        torch.cat(
                            [
                                finv.f_inv[:, i : i + 4, i : i + 4]
                                for i in range(0, finv.B, 4)
                            ],
                            dim=1,
                        )
                        .reshape((finv.d // finv.B, finv.B // 4, 4, 4))
                        .reshape((finv.d // 4, 4, 4))
                    )  # (Q, d/Q, Q) -> (d/Q, Q, Q)
                    block_finv_w[i] = torch.linalg.solve(
                        block_finv,
                        block_w,
                    )  # (d/Q, Q)
                    score = 0.5 * torch.einsum(
                        "bi,bi->b", block_w, block_finv_w[i]
                    )  # d/Q
                    scores[i] = (
                        score.unsqueeze(1)
                        .expand(-1, 4)
                        .reshape(self._params[i].data.shape)
                    )

            # make sure pruned ones will stay pruned
            for i, score in enumerate(scores):
                score[self._masks[i] == 0] = float("-inf")

        self._broadcast_list_from_main(scores)
        if self._mask_type == "block4":
            self._broadcast_list_from_main(block_finv_w)
            self._block_finv_w = block_finv_w  # cache for OBS weight update

        return scores

    @torch.no_grad()
    def pre_optim_step_update(self, masks: List[Tensor]):
        """
        Update the empirical inverse Fisher estimation based on the current gradients

        :param masks: latest masks that are applied to these parameters
        """
        if not self._enabled_grad_buffering:
            # only collect gradients when called during pruning step
            # this ignores calls invoked by manager during training
            return

        if self._finvs is None:
            self._setup_fisher_inverse(masks)

        for i, finv in enumerate(self._finvs):
            self._params[i].grad.mul_(masks[i])
            finv.add_grad(self._params[i].grad.view(-1).to(self._devices[i]))

    @torch.no_grad()
    def mask_update(self, masks: List[Tensor], mask_diffs: List[Tensor]):
        """
        Apply OBS weight update which zeros-out pruned weights and updates the
        remaining weights to preserve the loss.

        :param masks: latest masks to be applied to these parameters
        :param mask_diffs: mask diff values returned by mask_difference for these
            masks that describe how these masks changed since the last update
        """
        obs_updates = [None] * len(self._params)
        if self._is_main_proc:
            for i, param in enumerate(self._params):
                if self._mask_type == "unstructured":
                    obs_updates[i] = (
                        self._finvs[i]
                        .mul(
                            (param.data * (mask_diffs[i] == -1))
                            .view(-1)
                            .to(self._devices[i])
                            / (self._finvs[i].diag() + self._eps)
                        )
                        .view(param.data.shape)
                    )
                else:  # self._mask_type == "block4":
                    obs_updates[i] = (
                        self._finvs[i]
                        .mul(
                            self._block_finv_w[i].view(-1)
                            * (mask_diffs[i] == -1).view(-1).to(self._devices[i])
                        )
                        .view(param.data.shape)
                    )

        self._broadcast_list_from_main(obs_updates)
        # apply OBS update and manually zero-out pruned weights
        for i, param in enumerate(self._params):
            param.data -= obs_updates[i].to(param.data.device)
            param.data[mask_diffs[i] == -1] = 0.0

        self._finvs = None

    def _validate(self):
        if self._mask_type == "block4":
            for param in self._params:
                if param.numel() % self._fisher_block_size != 0:
                    raise ValueError(
                        "number of elements in each param must be divisible \
                        by fisher_block_size"
                    )

    def _setup_fisher_inverse(self, masks: List[Tensor]):
        self._masks = masks  # to be used by score_parameters
        self._finvs = []
        for i, param in enumerate(self._params):
            self._finvs.append(
                EmpiricalBlockFisherInverse(
                    self._num_grads,
                    self._fisher_block_size,
                    param.numel(),
                    self._damp,
                    self._devices[i],
                )
            )


class EmpiricalBlockFisherInverse:
    def __init__(
        self,
        num_grads: int,
        fisher_block_size: int,
        num_weights: int,
        damp: float,
        device: torch.device,
    ):
        self.m = num_grads
        self.B = fisher_block_size
        self.d = num_weights
        self.damp = damp
        self.dev = device

        self.num_blocks = math.ceil(self.d / self.B)
        self.f_inv = (
            (1.0 / self.damp * torch.eye(n=self.B, device=self.dev))
            .unsqueeze(0)
            .repeat(self.num_blocks, 1, 1)
        )  # O(d x B) memory

    def add_grad(self, g: Tensor):
        """
        Updates empirical Fisher inverse with a new gradient
        :param g: a collected gradient
        """
        # if 'd / B' is not integer, pad with zeros for batch calculations
        if g.numel() < self.num_blocks * self.B:
            g = torch.cat(
                [g, torch.zeros(self.num_blocks * self.B - g.numel(), device=g.device)]
            )

        # prepare grad for batch calculations
        g = g.view(self.num_blocks, self.B)

        # batched f_inv x g: (batch, B, B) x (batch, B) -> (batch, B)
        finv_g = torch.einsum("bij,bj->bi", self.f_inv, g)

        # scalar denominator for each batch: (batch)
        alpha = (self.m + torch.einsum("bi,bi->b", g, finv_g)).sqrt().unsqueeze(1)
        finv_g /= alpha

        # update f_inv with new outer product: (batch, B) x (batch, B) -> (batch, B, B)
        self.f_inv.baddbmm_(finv_g.unsqueeze(2), finv_g.unsqueeze(1), alpha=-1)

    def diag(self) -> Tensor:
        """
        :return: diagonal of the Fisher inverse matrix
        """
        return self.f_inv.diagonal(dim1=1, dim2=2).flatten()[: self.d]

    def mul(self, v: Tensor) -> Tensor:
        """
        Computes matrix-vector product of the Fisher inverse matrix and a vector
        :param v: a vector to compute matrix-vector product with
        :return: result of the matrix-vector multiplication
        """
        if v.numel() < self.num_blocks * self.B:
            v = torch.cat(
                [v, torch.zeros(self.num_blocks * self.B - v.numel(), device=v.device)]
            )
        return torch.bmm(
            self.f_inv, v.view(self.num_blocks, self.B).unsqueeze_(2)
        ).flatten()[: self.d]
