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
TODO add in future releases
"""
import math
import torch
import logging
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from typing import List, Optional, Union

from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.mask_creator import (
    PruningMaskCreator,
    get_mask_creator_default,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_base import (
    BaseGradualPruningModifier,
)

from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsGradScorer
from sparseml.pytorch.utils.logger import BaseLogger
from sparseml.pytorch.utils.helpers import tensor_density

from .spdy_utils import *
from .budget_counter_utils import *
from .pruning_handle import AdaOBCHandle


__all__ = [
    "SPDYPruningModifier",
    "SPDYPruningParamScorer",
]


_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class SPDYPruningModifier(BaseGradualPruningModifier):
    """

    | Sample yaml:
    |   !SPDYPruningModifier
    |       init_sparsity: 0.7
    |       final_sparsity: 0.9
    |       start_epoch: 2.0
    |       end_epoch: 26.0
    |       update_frequency: 4.0
    |       params: ["re:.*weight"]
    |       leave_enabled: True
    |       inter_func: cubic
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
        is supported. 
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
        mask_type: str = "unstructured",
        num_calibration_samples: int = 1024,
        dim_batch_size: int = 32,
        rel_damp: float = 0.0,
        handle_verbose: bool = False,
        spdy_verbose: bool = False,
        min_sparsity: float = 0.0,
        max_sparsity: float = 1.0,
        num_sparsity_levels: int = 40,
        budget_metric: str = 'flops',
        num_buckets: int = 10000,
        num_rand_inits: int = 100,
        resample_perc: float = 0.1, 
        patience: int = 100,
        device: str = 'cpu',
        save_profile: bool = False,
        save_profile_path: str = './best_coefs.npy',
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
        self._supported_masks = ("unstructured",)
        self._bugdet_metrics = ("params", "flops")
        # SPDY + OBC params
        self._spdy_kw = dict(
            num_calibration_samples=num_calibration_samples,
            dim_batch_size=dim_batch_size,
            rel_damp=rel_damp,
            handle_verbose=handle_verbose,
            spdy_verbose=spdy_verbose,
            min_sparsity=min_sparsity,
            max_sparsity=max_sparsity,
            num_sparsity_levels=num_sparsity_levels,
            budget_metric=budget_metric,
            num_buckets=num_buckets,
            num_rand_inits=num_rand_inits,
            resample_perc=resample_perc,
            patience=patience,
            device=device,
            save_profile=save_profile,
            save_profile_path=save_profile_path,
            store_on_drive=store_on_drive,
            store_dir=store_dir,
        )

        self._validate()


    def _validate(self):
        assert self._mask_type in self._supported_masks
        assert 0.0 <= self._spdy_kw["min_sparsity"] <= self._spdy_kw["max_sparsity"] < 1.0
        assert self._spdy_kw["budget_metric"] in self._bugdet_metrics


    @ModifierProp()
    def mask_type(self) -> str:
        """
        :return: the mask type used
        """
        return self._mask_type


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
        :param loss_fn: loss function needed for model evaluation
        :param loader: Dataloader for work with caliration
        :param epoch: the epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: optional list of loggers to log the modification process to

        :param kwargs: optional kwargs to support specific arguments
            for individual modifiers.
        """
        _LOGGER.info("Initializing SPDYPruningModifier")

        if 'loss_fn' not in kwargs and 'loader' not in kwargs:
            raise RuntimeError("loss_fn and loader has to provided for SPDY+OBC")
        else:
            self._model = module
            self._loader = kwargs["loader"]
            self._loss_fn = kwargs["loss_fn"]

        super().initialize(module, epoch, loggers, **kwargs)


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

        return SPDYPruningParamScorer(
            model=self._model,
            loader=self._loader,
            loss_fn=self._loss_fn,
            params=params,
            layers=layers,
            layer_names=layer_names,
            **self._spdy_kw
        )


    def check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        if steps_per_epoch == 1 and not math.isinf(epoch):
            return  # not a one-shot run

        _LOGGER.info("Running SPDY+OBC Pruning")
        if self._scorer._is_main_proc:
            self._scorer._enabled_spdy_preparation = True
        
        # get mean sparsity per all modules
        mean_sparsity = np.mean(self.get_applied_sparsity_for_epoch(epoch, 1))
        self._scorer.update_target(mean_sparsity)

        is_training = module.training
        module.eval()
        self._module_masks.pre_optim_step_update()
        self._pre_step_completed = True
        if is_training:
            _LOGGER.debug("Setting the model back to the train mode")
            module.train()

        if self._scorer._is_main_proc:
            self._scorer._enabled_spdy_preparation = False

        super().check_mask_update(module, epoch, steps_per_epoch, **kwargs)


class SPDYPruningParamScorer(PruningParamsGradScorer):

    def __init__(
        self,
        model: Module,
        loader: DataLoader,
        loss_fn: Module,
        layers: List[Module],
        layer_names: List[str],
        params: List[Parameter],
        num_calibration_samples: int = 1024,
        dim_batch_size: int = 32,
        rel_damp: float = 0.0,
        handle_verbose: bool = False,
        spdy_verbose: bool = False,
        min_sparsity: float = 0.0,
        max_sparsity: float = 1.0,
        num_sparsity_levels: int = 40,
        budget_metric: str = 'flops',
        num_buckets: int = 10000,
        num_rand_inits: int = 100,
        patience: int = 100,
        resample_perc: float = 0.1, 
        device: str = 'cpu',
        save_profile: bool = False,
        save_profile_path: str = './best_coefs.npy',
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
        self._num_calibration_samples = num_calibration_samples
        self._dim_batch_size = dim_batch_size
        self._rel_damp = rel_damp
        self._spdy_verbose = spdy_verbose
        self._handle_verbose = handle_verbose
        self._min_sparsity = min_sparsity
        self._max_sparsity = max_sparsity
        self._num_sparsity_levels = num_sparsity_levels
        self._budget_metric = budget_metric
        self._num_buckets = num_buckets
        self._num_rand_inits = num_rand_inits
        self._patience = patience
        self._resample_perc = resample_perc
        self._device = device
        self._save_profile = save_profile
        self._save_profile_path = save_profile_path
        self._store_on_drive = store_on_drive
        self._store_dir = store_dir

        # init OBS handles
        self.obs_handles: List[AdaOBCHandle] = [None] * len(self._params)
        for layer_id, layer in enumerate(layers):
            # add obs handle to each module
            self.obs_handles[layer_id] = AdaOBCHandle(
                layer,
                num_samples=num_calibration_samples,
                dim_batch_size=dim_batch_size,
                rel_damp=rel_damp,
                verbose=handle_verbose
            )
        # make sparsity levels
        l_ = np.log2(1.0 - min_sparsity)
        r_ = np.log2(1.0 - max_sparsity)
        self.sparsities = 1 - np.logspace(l_, r_, num=num_sparsity_levels, base=2)
        # init weight database
        self._weight_database = None
        self._errs_per_layer = None
        self._budgets_per_layer = None
        self._enabled_spdy_preparation = False
        self._cur_target = 1.0

    
    def collect_hessians(self):
        def update_H_hook(layer_id: int):
            def _hook(layer, inp, out):
                self.obs_handles[layer_id].update_H(inp[0].data)
            return _hook

        # register batch collecting hook
        hooks = [None] * len(self._params)
        for layer_id, obs_handle in enumerate(self.obs_handles):
            layer = obs_handle.layer
            hooks[layer_id] = layer.register_forward_hook(update_H_hook(layer_id))
        # collect batches
        with torch.no_grad():
            for inputs, _ in self._loader:
                inputs = inputs.to(self._device)
                _ = self._model(inputs)
        # remove hooks
        for hook in hooks:
            hook.remove()


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


    @torch.no_grad()
    def pre_optim_step_update(self, masks: List[Tensor]):
        if not self._enabled_spdy_preparation:
            return 
        # collect hessians
        _LOGGER.info("Collecting hessians...")
        self.collect_hessians()
        # prepare losses and traces
        for layer_name, obs_handle in self.obs_handles.items():
            obs_handle.prepare_losses_and_traces()
        # create weight database
        _LOGGER.info("Creating weight database...")
        self._weight_database = WeightDatabase(
            store_on_drive=self._store_on_drive,
            store_dir=self._store_dir
        )
        for layer_name, obs_handle in zip(self._layer_names, self.obs_handles):
            self._weight_database[layer_name] = obs_handle.get_pruning_database(self.sparsities)
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


    def collect_errors(self) -> None:
        # reinit errs
        self._errs_per_layer = {
            layer_name: np.zeros_like(self.sparsities)
            for layer_name, _ in self.obs_handles.items()
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
        scores = [None] * len(self._params)

        if self._is_main_proc:
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
