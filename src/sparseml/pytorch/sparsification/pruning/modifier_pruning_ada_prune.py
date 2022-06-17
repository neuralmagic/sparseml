#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Modifiers classes related to magnitude pruning
"""
from copy import deepcopy
import re
import time
import torch
import logging
import numpy as np
import torch.nn.functional as F

from functools import partial
from typing import Dict, List, Union, Optional

from torch import Tensor
from torch.nn import Module

from sparseml.pytorch.utils.logger import BaseLogger
from sparseml.pytorch.sparsification.modifier import PyTorchModifierYAML

from sparseml.pytorch.sparsification.pruning.modifier_pruning_base import \
    BaseGradualPruningModifier
from sparseml.pytorch.sparsification.pruning.modifier_pruning_magnitude import \
    MagnitudePruningModifier, GlobalMagnitudePruningModifier
from sparseml.pytorch.sparsification.pruning.modifier_pruning_mfac import \
    MFACPruningModifier
from sparseml.pytorch.sparsification.pruning.modifier_pruning_movement import \
    MovementPruningModifier


_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)


__all__ = [
    "AdaPruneMagnitudeModifier",
    "AdaPruneGlobalMagnitudeModifier",
    "AdaPruneMFACModifier",
    "AdaPruneMovementModifier",
]


@PyTorchModifierYAML()
class AdaPruneBaseModifier(BaseGradualPruningModifier):

    def __init__(
        self,
        init_sparsity: Union[float, str],
        final_sparsity: Union[float, Dict[float, List[str]]],
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        modules: Union[str, List[str]],
        num_calibration_samples: int,
        num_calibration_steps: int,
        calibration_batch_size: int,
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        mask_type: str = "unstructured",
        num_recompute: int = 1,
        calibration_lr: float = 0.1,
        calibration_momentum: float = 0.9,
        calibration_device: str = 'cpu',
        calibration_log_freq: int = -1,
        **kwargs
    ):
        super(AdaPruneBaseModifier, self).__init__(
            params=params,
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            inter_func=inter_func,
            mask_type=mask_type,
            leave_enabled=leave_enabled
        )
        # modules whose activations will be needed
        self._modules = modules
        # ada prune params
        self._num_recompute = num_recompute
        self._num_calibration_steps = num_calibration_steps
        self._num_calibration_samples = num_calibration_samples
        self._calibration_batch_size = calibration_batch_size
        self._calibration_lr = calibration_lr
        self._calibration_momentum = calibration_momentum
        self._calibration_device = calibration_device
        self._calibration_log_freq = calibration_log_freq
        # previous sparsity
        self._prev_sparsity = None
        # samples used for calibration step
        self._calibration_samples = None
        # loader used for calibration step
        self._calibration_loader =  None


    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        teacher_module: Optional[Module] = None,
        **kwargs,
    ):
        """
        Grab the layers and apply if epoch in range to control pruning for.
        If `grad_sampler: GradSampler` is present in kwargs, then will add
        it to this class and use the sampler instead of live gradient buffering

        :param module: the PyTorch model/module to modify
        :param epoch: The epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: Optional list of loggers to log the modification process to
        :param kwargs: Optional kwargs to support specific arguments
            for individual modifiers.
        :param teacher_model: the tea
        """

        # init calibration loader
        if not "calibration_loader" in kwargs:
            raise RuntimeError("No loader with calibration images provided")
        else:
            self._calibration_loader = kwargs["calibration_loader"]

        # if teacher model not provided use the copy of the modified Module
        if teacher_module is not None:
            self._teacher_module = teacher_module
        else: 
            _LOGGER.info("Using the copy of the provided module as teacher")
            self._teacher_module = deepcopy(module)

        # init dict of sparsified modules for further use
        self._submodule_dict = {}
        for submodule_name, submodule in module.named_modules():
            if re.search(self._modules, submodule_name):
                self._submodule_dict[submodule_name] = submodule
        # init optimizer for each module
        self._optimizer_dict = {
            submodule_name : torch.optim.SGD(
                submodule.parameters(), 
                lr=self._calibration_lr, 
                momentum=self._calibration_momentum
            ) 
            for submodule_name, submodule in self._submodule_dict.items()
        }
        # collect calibration images
        self._collect_calibration_images()
        # init parent class
        super().initialize(module, epoch, loggers, **kwargs)


    def _collect_calibration_images(self):
        num_collected = 0
        self._calibration_samples = []
        for samples in self._calibration_loader:
            # update list of collected samples
            self._calibration_samples.append(
                samples[:(self._num_calibration_samples - num_collected)])
            # update number of collected samples
            num_collected += len(samples)
            if num_collected == self._num_calibration_samples:
                break
        # concatenate batches and put on device
        self._calibration_samples = torch.cat(
            self._calibration_samples).to(self._calibration_device)
        _LOGGER.info("Calibration images collected.")

    
    def calibrate(self):
        # add activation hooks for teacher model
        hooks = {}
        for submodule_name, teacher_submodule in self._teacher_module.named_modules():
            if re.search(self._modules, submodule_name):
                def save_input_output(mod, inp, out, submodule_name=''):
                    module_inputs[submodule_name] = inp[0]
                    module_outputs[submodule_name] = out
                # add hook storing activations
                hooks[submodule_name] = teacher_submodule.register_forward_hook(
                    partial(save_input_output, submodule_name=submodule_name))

        # make list of module masks
        module_masks = {}
        for submodule_name, submodule in self._submodule_dict.items():
            if re.search(self._modules, submodule_name):
                module_masks[submodule_name] = (submodule.weight != 0)

        # awful, but works
        module_device = next(iter(self._teacher_module.parameters()))[0].device

        # make list of all activation ids
        calibration_ids = np.arange(len(self._calibration_samples))
        # optimize per layer
        for step in range(self._num_calibration_steps):
            # reinit activations
            module_inputs  = {}
            module_outputs = {}
            # get batch idx
            batch_idx = np.random.choice(calibration_ids, size=self._calibration_batch_size, replace=False)
            with torch.no_grad():
                _ = self._teacher_module(self._calibration_samples[batch_idx].to(module_device))
            # optimize separately each module
            for submodule_name, submodule in self._submodule_dict.items():
                optimizer = self._optimizer_dict[submodule_name]
                # get module mask
                module_mask = module_masks[submodule_name]
                # get inputs and targets
                inputs  = module_inputs[submodule_name]
                targets = module_outputs[submodule_name]
                # get predict   ions
                preds = submodule(inputs)
                # get loss
                loss = F.mse_loss(preds, targets)
                # make gradient step  
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                with torch.no_grad():
                    submodule.weight *= module_mask
            if step % self._calibration_log_freq == 0 and self._calibration_log_freq > 0:
                _LOGGER.info(f"AdaPrune Calibration Step [{step}/{self._num_calibration_steps}]")
        # remove hooks
        for _, hook in hooks.items():
            hook.remove()


    # OVERRIDEN
    def check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        """
        Update mask values if necessary

        :param module: module to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        started = self.started
        if self.start_pending(epoch, steps_per_epoch):
            self._module_masks.enabled = True
            started = True

        if not self._pre_step_completed:
            # do pre optim step before mask update on update steps
            self._module_masks.pre_optim_step_update()
            self._pre_step_completed = True

        if started:
            # get sparsity level to be applied
            self._applied_sparsity = self.get_applied_sparsity_for_epoch(
                epoch, steps_per_epoch
            )

            # torch tensor for vectorized operations
            _applied_sparsity = torch.tensor(self._applied_sparsity)
            if self._prev_sparsity is None:
                _prev_sparsity = torch.zeros_like(_applied_sparsity)
            else:
                _prev_sparsity = torch.tensor(self._prev_sparsity)

            # compute sparsity difference on step
            _sparsity_diff = (_applied_sparsity - _prev_sparsity) / self._num_recompute

            _start = time.perf_counter()
            for i in range(self._num_recompute):
                _cur_sparsity = (_prev_sparsity + (i + 1) * _sparsity_diff).tolist()
                self._module_masks.update_param_masks(target=_cur_sparsity)
                self.calibrate()
            _end = time.perf_counter()
            _LOGGER.info(
                f'{self._num_recompute} AdaPrune recomputations steps took {(_end - _start):.3f} s.')

            self._sparsity_applied = True
            self._prev_sparsity = self._applied_sparsity

        if self.end_pending(epoch, steps_per_epoch):
            self._module_masks.pruning_end(self._leave_enabled)


@PyTorchModifierYAML()
class AdaPruneMagnitudeModifier(AdaPruneBaseModifier, MagnitudePruningModifier):
    """
    This subclass of the AdaPruneBaseModifier implements 
    AdapPrune training with the MagnitudePruning scorer as 
    in the original paper https://arxiv.org/pdf/2106.12379.pdf. 
    """
    pass


@PyTorchModifierYAML()
class AdaPruneGlobalMagnitudeModifier(AdaPruneBaseModifier, GlobalMagnitudePruningModifier):
    """
    This subclass of the AdaPruneBaseModifier implements 
    AdapPrune training with the GlobalMagnitudePruningModifier scorer.
    """
    pass


@PyTorchModifierYAML()
class AdaPruneMovementModifier(AdaPruneBaseModifier, MovementPruningModifier):
    """
    This subclass of the AdaPruneBasePruningModifier implements 
    AdapPrune training with the MovementPruningModifier scorer.
    """
    pass


#TODO add GradSampler support
@PyTorchModifierYAML()
class AdaPruneMFACModifier(AdaPruneBaseModifier, MFACPruningModifier):


    def __init__(
        self,
        init_sparsity: Union[float, str],
        final_sparsity: Union[float, Dict[float, List[str]]],
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        modules: Union[str, List[str]],
        num_calibration_samples: int,
        num_calibration_steps: int,
        calibration_batch_size: int,
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        mask_type: str = "unstructured",
        num_recompute: int = 1,
        calibration_lr: float = 0.1,
        calibration_momentum: float = 0.9,
        calibration_device: str = 'cpu',
        use_gradient_buffering: Optional[bool] = None,
        num_grads: Union[Dict[float, int], int] = 64,
        damp: float = 1e-5,
        grads_device: Union[str, int] = "cpu",
        fisher_block_size: int = 2000,
        num_pages: int = 1,  # break computation into pages when block size is None
        available_devices: Optional[List[str]] = None,
        **kwargs
    ):
        super(AdaPruneMFACModifier, self).__init__(
            params=params,
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            inter_func=inter_func,
            mask_type=mask_type,
            leave_enabled=leave_enabled,
            modules=modules,
            num_calibration_samples=num_calibration_samples,
            num_calibration_steps=num_calibration_steps,
            calibration_batch_size=calibration_batch_size,
            num_recompute=num_recompute,
            calibration_lr=calibration_lr,
            calibration_momentum=calibration_momentum,
            calibration_device=calibration_device,
            use_gradient_buffering=use_gradient_buffering,
            num_grads=num_grads,
            damp=damp,
            grads_device=grads_device,
            fisher_block_size=fisher_block_size,
            num_pages=num_pages,
            available_devices=available_devices
        )