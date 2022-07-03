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
import math
import torch
import logging
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Union

from sparseml.utils import interpolate
from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.mask_creator import (
    PruningMaskCreator,
    get_mask_creator_default,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_base import (
    BaseGradualPruningModifier,
)
from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsGradScorer
from sparseml.pytorch.utils import GradSampler, tensor_sparsity
from sparseml.pytorch.utils.logger import BaseLogger
from sparseml.pytorch.utils.helpers import tensor_density
# spdy imports
from .pruning_handle import FisherOBCHandle
# obs imports
from .modifier_pruning_obs import EmpiricalBlockFisherInverse
from .spdy_utils import *
from .budget_counter_utils import *


__all__ = [
    "SPDY_OBS_PruningModifier",
    "SPDY_OBS_PruningParamsScorer",
]


_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class SPDY_OBS_PruningModifier(BaseGradualPruningModifier):
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
    """

    _supported_masks = ("unstructured",)

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
        mask_type: str = "unstructured",
        num_grads: int = 1024,
        damp: float = 1e-7,
        fisher_block_size: int = 64,
        grad_sampler_kwargs: Dict[str, Any] = {},
        num_recomputations: int = 1,
        recomputation_inter_func: str = 'linear',
        obc_batch_size: int = 32,
        # SPDY kwargs
        spdy_verbose: bool = False,
        min_sparsity_level: float = 0.0,
        max_sparsity_level: float = 1.0,
        num_sparsity_levels: int = 40,
        num_buckets: int = 10000,
        num_rand_inits: int = 100,
        resample_perc: float = 0.1, 
        patience: int = 100,
        save_profile: bool = False,
        save_profile_path: str = './best_profile.npy',
        store_on_drive: bool = False,
        store_dir: str = '',

    ):
        super().__init__(
            params=params,
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            inter_func=inter_func,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            global_sparsity=True,
            leave_enabled=leave_enabled,
            parent_class_kwarg_names=[],
        )
        self._mask_type = mask_type
        self._num_grads = num_grads
        self._damp = damp
        self._fisher_block_size = fisher_block_size
        self._grad_sampler_kwargs = grad_sampler_kwargs
        self._num_recomputations = num_recomputations
        self._obc_batch_size = obc_batch_size
        self._grad_sampler = None
        self._recomputation_inter_func = recomputation_inter_func
        self._last_applied_sparsity = init_sparsity
        # SPDY kwargs
        self._spdy_kw=dict(
            spdy_verbose=spdy_verbose,
            min_sparsity_level=min_sparsity_level,
            max_sparsity_level=max_sparsity_level,
            num_sparsity_levels=num_sparsity_levels,
            num_buckets=num_buckets,
            num_rand_inits=num_rand_inits,
            resample_perc=resample_perc,
            patience=patience,
            save_profile=save_profile,
            save_profile_path=save_profile_path,
            store_on_drive=store_on_drive,
            store_dir=store_dir,
        )
        # check arguments
        self._validate()

    def _validate(self):
        if isinstance(self._damp, str):  # to support 'damp: 1e-7' in the recipe
            self._damp = float(self._damp)

        assert (
            self._mask_type in self._supported_masks
        ), f"{self._mask_type} mask_type not supported"

        assert self._recomputation_inter_func in ("linear", "cubic")


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
    def recomputation_inter_func(self) -> str:
        """
        :return: whether the nonzero weights are updated in OBS step
        """
        return self._recomputation_inter_func

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
        _LOGGER.info("Initializing SPDY+OBC PruningModifier")
        if (
            "grad_sampler" not in kwargs
            or "data_loader_builder" not in kwargs["grad_sampler"]
            or "loss_function" not in kwargs["grad_sampler"]
            or "calibration_loader" not in kwargs
            or "loss_fn" not in kwargs
        ):
            raise RuntimeError(
                "grad_sampler dict with data_loader_builder and loss_function "
                "must be provided to initialize GradSampler"
                "calibration loader and loss_fn"
                "must be provided for SPDY evalutation"
            )
        self._model   = module
        self._loader  = kwargs["calibration_loader"]
        self._loss_fn = kwargs["loss_fn"]

        if math.isinf(epoch): # hack to enable oneshot
            self._grad_sampler = GradSampler(
                kwargs["grad_sampler"]["data_loader_builder"](
                    **self._grad_sampler_kwargs
                ),
                kwargs["grad_sampler"]["loss_function"],
            )

        super().initialize(module, epoch, loggers, **kwargs)

        # if self._scorer._is_main_proc:  # grads collected only in the main proc
        self._grad_sampler = GradSampler(
            kwargs["grad_sampler"]["data_loader_builder"](
                **self._grad_sampler_kwargs
            ),
            kwargs["grad_sampler"]["loss_function"],
        )

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
        # extract layer names
        named_layers_and_params = self._create_named_layers_and_params(self._model)
        layers = [nlp.layer for nlp in named_layers_and_params]
        layer_names = [nlp.layer_name for nlp in named_layers_and_params]

        return SPDY_OBS_PruningParamsScorer(
            model=self._model,
            loader=self._loader,
            loss_fn=self._loss_fn,
            params=params,
            layers=layers,
            layer_names=layer_names,
            num_grads=self._num_grads,
            damp=self._damp,
            fisher_block_size=self._fisher_block_size,
            mask_type=self._mask_type,
            **self._spdy_kw
        )

    
    def _prepare(self,  module: Module):
        if self._scorer._is_main_proc:
            # collect grads for empirical inverse Fisher estimation
            self._scorer._enabled_grad_buffering = True

        self._collect_grad_samples(module, self._grad_sampler)
        self._pre_step_completed = True

        if self._scorer._is_main_proc:
            self._scorer._enabled_grad_buffering = False


    def check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        if steps_per_epoch == 1 and not math.isinf(epoch):
            return  # not a one-shot run

        if self._scorer._is_main_proc:
            _LOGGER.info("Running SPDY+OBS Pruning")
        # set here to prevent inf loop when super().check_mask_update(...)
        # is called num_recomputations times
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
            if self._scorer._is_main_proc:
                _LOGGER.info(f"Recomputation [{i}/{self._num_recomputations}]")
            # prepare for pruning
            self._prepare(module)
            recomputation_sparsity = [
                interpolate(
                    i,
                    0,
                    self._num_recomputations,
                    y0=start_sparsity,
                    y1=target_sparsity,
                    inter_func=self._recomputation_inter_func
                )
                for start_sparsity, target_sparsity in zip(last_applied_sparsities, to_apply_sparsities)
            ]
            # update scorer target sparsity
            self._scorer.update_target(np.mean(recomputation_sparsity))
            # overwrite sparsity targets when there are recomputations
            super().check_mask_update(
                module,
                epoch,
                steps_per_epoch,
                recomputation_sparsity=recomputation_sparsity,
            )

        self._last_applied_sparsity = to_apply_sparsities
            

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


class SPDY_OBS_PruningParamsScorer(PruningParamsGradScorer):
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
        model: Module,
        loader: DataLoader,
        loss_fn: Module,
        layers: List[Module],
        layer_names: List[str],
        params: List[Parameter],
        num_grads: int,
        damp: float,
        fisher_block_size: int,
        mask_type: str,
        obc_batch_size: int = 32,
        spdy_verbose: bool = False,
        min_sparsity_level: float = 0.0,
        max_sparsity_level: float = 1.0,
        num_sparsity_levels: int = 40,
        budget_metric: str = 'params',
        num_buckets: int = 10000,
        num_rand_inits: int = 100,
        patience: int = 100,
        resample_perc: float = 0.1, 
        save_profile: bool = False,
        save_profile_path: str = './best_profile.npy',
        store_on_drive: bool = False,
        store_dir: str = '',
    ):
        super().__init__(params)
        # set initial params
        self._model = model
        self._loader = loader
        self._layers = layers
        self._loss_fn = loss_fn
        self._layer_names = layer_names
        # Fisher params
        self._damp = damp
        self._num_grads = num_grads
        self._fisher_block_size = fisher_block_size
        self._mask_type = mask_type
        # SPDY params
        self._spdy_verbose = spdy_verbose
        self._min_sparsity_level = min_sparsity_level
        self._max_sparsity_level = max_sparsity_level
        self._num_sparsity_levels = num_sparsity_levels
        self._budget_metric = budget_metric
        self._num_buckets = num_buckets
        self._num_rand_inits = num_rand_inits
        self._patience = patience
        self._resample_perc = resample_perc
        self._save_profile = save_profile
        self._save_profile_path = save_profile_path
        self._store_on_drive = store_on_drive
        self._store_dir = store_dir
        self._num_calibration_samples = len(self._loader.dataset)

        self._Finvs: List[EmpiricalBlockFisherInverse] = None
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
        # may lead to failure in some cases (but hopefull not)
        self._device = self._params[0].device

        self._pickle_exclude_params.extend(
            [
                "_Finvs",
                "_enabled_grad_buffering",
                "_devices",
            ]
        )

        # init OBS handles
        self.obc_handles: List[FisherOBCHandle] = [None] * len(params)
        for i, param in enumerate(params):
            # add obs handle to each module
            self.obc_handles[i] = FisherOBCHandle(
                param,
                obc_batch_size=obc_batch_size,
                verbose=False
            )

        # make sparsity levels
        l_ = np.log2(1.0 - min_sparsity_level)
        r_ = np.log2(1.0 - max_sparsity_level)
        self.sparsities = 1 - np.logspace(l_, r_, num=num_sparsity_levels, base=2)
        # init weight database
        self._weight_database = None
        self._errs_per_layer = None
        self._budgets_per_layer = None
        self._enabled_spdy_preparation = False
        self._cur_target = 1.0

    def _setup_FisherInverse(self, masks: List[Tensor]):
        self._masks = masks  # to be used by score_parameters
        self._Finvs = []
        for i, param in enumerate(self._params):
            self._Finvs.append(
                EmpiricalBlockFisherInverse(
                    self._num_grads,
                    self._fisher_block_size,
                    param.numel(),
                    self._damp,
                    self._devices[i],
                )
            )


    def compute_budget(self):
        if self._budget_metric == 'params':
            # param_counts
            budgets_dense = get_param_counter(self._layer_names, self._layers)
        if self._budget_metric == 'flops':
            sample_input, _ = next(iter(self._loader))
            sample_input = sample_input.to(self._device)
            budgets_dense = get_flop_counter(self._model, self._layer_names, self._layers, sample_input)

        budgets_per_sparsity = {}
        for layer_name in self._weight_database.keys():
            budgets_per_sparsity[layer_name] = [
                int(budgets_dense[layer_name] * tensor_density(self._weight_database.get(layer_name, i)).item())
                for i in range(self._num_sparsity_levels)
            ]

        return budgets_per_sparsity

    
    def update_target(self, target_sparsity: float):
        self._cur_target = 1 - target_sparsity


    def collect_errors(self) -> None:
        # reinit errs
        self._errs_per_layer = {
            layer_name: np.zeros_like(self.sparsities)
            for layer_name in self._layer_names
        }
        # register batch collecting hook
        hooks = {}

        def accum_err_inp_out_hook(layer_name):
            def _hook(layer, inp, out):
                weight = layer.weight
                hooks[layer_name].remove()
                for i, _ in enumerate(self.sparsities):
                    weight.data = self._weight_database.get(layer_name, i)
                    self._errs_per_layer[layer_name][i] += \
                        (len(inp) / self._num_calibration_samples) * F.mse_loss(layer(inp[0]), out)
                # restore original weight
                weight.data = self._weight_database.get(layer_name, 0)
            return _hook

        for layer_name, layer in zip(self._layer_names, self._layers):
            hooks[layer_name] = layer.register_forward_hook(accum_err_inp_out_hook(layer_name))

        # collect batches (hooks are removed automatically)
        with torch.no_grad():
            for inputs, _ in self._loader:
                inputs = inputs.to(self._device)
                _ = self._model(inputs)
    

    @torch.no_grad()
    def score_parameters(self) -> List[Tensor]:
        """
        :return: List of Tensors the same shapes as the given Parameters where
            each Parameter's elements are scored based on the blockwise OBS
        """
        scores = [None] * len(self._params)

        if self._is_main_proc:
            # prepare losses and traces
            for i, obc_handle in enumerate(self.obc_handles):
                # set fisher inverse
                obc_handle.set_Finv(self._Finvs[i].F_inv)
                # compute losses and weight traces
                obc_handle.prepare_losses_and_traces()

            _LOGGER.info("Creating weight database...")
            self._weight_database = WeightDatabase(
                store_on_drive=self._store_on_drive,
                store_dir=self._store_dir
            )
            for layer_name, obc_handle in zip(self._layer_names, self.obc_handles):
                self._weight_database[layer_name] = obc_handle.get_pruning_database(self.sparsities)
            # restore weights (to the one before pruning)
            for layer_name in self._weight_database.keys():
                layer = self._model.get_submodule(layer_name)
                layer.weight.data = self._weight_database.get(layer_name, 0)
            # dict of errors per layer and sparsity
            _LOGGER.info("Collecting errors per layer...")
            self.collect_errors()
            # compute budgets
            _LOGGER.info("Computation of budgets...")
            self._budgets_per_layer = self.compute_budget()

            spdy_solver = SPDY(
                self._model,
                self._loader,
                self._loss_fn,
                self._weight_database,
                self._errs_per_layer,
                self._budgets_per_layer,
                target_budget_frac=self._cur_target,
                num_buckets=self._num_buckets, 
                num_rand_inits=self._num_rand_inits,
                resample_perc=self._resample_perc,
                patience=self._patience,
                device=self._device,
                verbose=self._spdy_verbose,
                save_profile=self._save_profile,
                save_profile_path=self._save_profile_path
            )

            spdy_solver.search()
            # best solution found
            self._best_solution = spdy_solver.best_solution
            # hack to score model according to SPDY
            for layer_id, layer_name in enumerate(self._layer_names):
                weight = self._weight_database.get(layer_name, self._best_solution[layer_id])
                scores[layer_id] = (weight != 0).to(torch.float32)

        self._broadcast_list_from_main(scores)

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

        if self._Finvs is None:
            self._setup_FisherInverse(masks)

        for i, finv in enumerate(self._Finvs):
            self._params[i].grad.mul_(masks[i])
            finv.add_grad(self._params[i].grad.view(-1).to(self._devices[i]))


    @torch.no_grad()
    def mask_update(self, masks: List[Tensor], mask_diffs: List[Tensor]):
        '''
        Set the weights from the chosen SPDY profile
        '''
        # collect weights chosen by SPDY
        spdy_weights = [None] * len(self._params)
        if self._is_main_proc:
            for i, (layer_name, sp_lvl) in enumerate(zip(self._layer_names, self._best_solution)):
                spdy_weights[i] = self._weight_database.get(layer_name, sp_lvl)

        self._broadcast_list_from_main(spdy_weights)

        for i, param in enumerate(self._params):
            param.data = spdy_weights[i].to(param.device)

        # clean-up
        if self._is_main_proc:
            self._weight_database = None
