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

import math
from typing import List, Union

from torch.nn import Module, Parameter
from torch.optim.optimizer import Optimizer

from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.mask_creator import (
    PruningMaskCreator,
    get_mask_creator_default,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_base import (
    BasePruningModifier,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_magnitude import (
    MagnitudePruningParamsScorer,
)
from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsScorer


__all__ = ["ACDCPruningModifier"]


@PyTorchModifierYAML()
class ACDCPruningModifier(BasePruningModifier):
    """
    Implementation of
    Alternating Compressed/DeCompressed Training of Deep Neural Networks:
    https://arxiv.org/pdf/2106.12379.pdf.
    AC/DC performs co-training of sparse and dense models, and can return both an
    accurate sparse model, and a dense model.
    | Sample yaml:
    |   !ACDCPruningModifier
    |       compression_sparsity: 0.9
    |       start_epoch: 0
    |       end_epoch: 100
    |       update_frequency: 5
    |       params: __ALL_PRUNABLE__
    |       global_sparsity: True

    :param compression_sparsity: The sparsity enforced during the compression phase.
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The length (in epochs) of compression/decompression phase
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights. If a sparsity to param mapping is defined by
        final_sparsity, then params should be set to []
    :param global_sparsity: set True to enable global pruning. if False, pruning will
        be layer-wise. Default is True
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune. Default is True
    :param mask_type: String to define type of sparsity to apply. May be 'unstructred'
        for unstructured pruning or 'block4' for four block pruning or a list of two
        integers for a custom block shape. Default is 'unstructured'
    :param momentum_buffer_reset: set True to reset momentum buffer
        before algorithm enters a consecutive decompression phase.
        According to the paper:
            "once all weights are re-introduced, it is beneficial
            to reset to 0 the gradient momentum term of the optimizer;
            this is particularly useful for the weights
            that were previously pruned, which would otherwise
            have stale versions of gradients."
        Default is True
    """

    def __init__(
        self,
        compression_sparsity: float,
        start_epoch: Union[int, float],
        end_epoch: Union[int, float],
        update_frequency: Union[int, float],
        params: Union[str, List[str]],
        global_sparsity: bool = True,
        leave_enabled: bool = True,
        momentum_buffer_reset: bool = True,
        mask_type: str = "unstructured",
    ):
        # AC/DC assumes that variables `start_epoch`, `end_epoch`
        # and `update_frequency` are integers.
        start_epoch = self._assert_is_integer(start_epoch)
        end_epoch = self._assert_is_integer(end_epoch)
        update_frequency = self._assert_is_integer(update_frequency)

        # because method does not involve any interpolation
        # compression sparsity (final sparsity) is a single float.
        self._compression_sparsity = compression_sparsity
        self._decompression_sparsity = 0.0  # this is implicitly assumed in paper.
        self._is_phase_decompression = True
        self._num_phase = None
        self._momentum_buffer_reset = momentum_buffer_reset
        self._mask_type = mask_type

        super(ACDCPruningModifier, self).__init__(
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            global_sparsity=global_sparsity,
            params=params,
            leave_enabled=leave_enabled,
        )

        self._momentum_buffer_empty = True

    def optimizer_post_step(
        self, module: Module, optimizer: Optimizer, epoch: float, steps_per_epoch: int
    ):
        """
        Reapply the mask after the optimizer step in case the optimizer
        has momentum that may have moved weights from 0. Additionally,
        reset momentum buffer if new dense phase starts.

        :param module: module to modify
        :param optimizer: optimizer to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        super().optimizer_post_step(module, optimizer, epoch, steps_per_epoch)

        # be sure to apply mask again after optimizer update because
        # weights may have changed (optimizer with momentum, not masking gradient)

        if (
            self._momentum_buffer_reset
            and self._is_phase_decompression
            and not self._momentum_buffer_empty
        ):
            """
            This condition is only evaluated when `momentum_buffer_reset`
            strategy is True. When entering decompression phase check
            whether momentum buffer is empty. If not
            (this happens always before the first epoch of the decompression phase),
            reset!
            """
            self._reset_momentum_buffer(optimizer)
            self._momentum_buffer_empty = True

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
        if epoch == float("inf"):
            return self._compression_sparsity

        self._num_phase = math.floor((epoch - self.start_epoch) / self.update_frequency)

        if self._num_phase % 2 == 0 and self.end_epoch - self.update_frequency > epoch:
            # entering decompression phase
            self._is_phase_decompression = True
            applied_sparsity = self._decompression_sparsity
        else:
            # entering compression phase
            self._is_phase_decompression = False
            applied_sparsity = self._compression_sparsity
            # flag denoting that the momentum buffer
            # is non-zero in compression phase.
            self._momentum_buffer_empty = False

        return applied_sparsity

    @ModifierProp()
    def mask_type(self) -> str:
        """
        :return: the mask type used
        """
        return self._mask_type

    @ModifierProp(serializable=True)
    def momentum_buffer_reset(self) -> bool:
        """
        :return: True to reset the gradient momentum
                 (momentum buffer) term of the optimizer
                 to zero before every decompression phase.
        """
        return self._momentum_buffer_reset

    @momentum_buffer_reset.setter
    def momentum_buffer_reset(self, value: bool):
        """
        :param value: whether we use momentum buffer reset strategy or not.
        """
        self._momentum_buffer_reset = value

    @ModifierProp(serializable=True)
    def compression_sparsity(self) -> float:
        """
        :return: The sparsity enforced during the compression phase.
        """
        return self._compression_sparsity

    def _get_scorer(self, params: List[Parameter]) -> PruningParamsScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
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

    @staticmethod
    def _reset_momentum_buffer(optimizer):
        if "state" in optimizer.state_dict():
            for param_buffer in optimizer.state_dict()["state"].values():
                if "momentum_buffer" not in param_buffer:
                    continue
                param_buffer["momentum_buffer"].mul_(0.0)

    @staticmethod
    def _assert_is_integer(x):
        """
        Check if x, which is expected to be either a float or int,
        can be evaluated as int.
        If True, return and integer, else, raise ValueError.

        :param x: an integer or a float.
        :return: an integer
        """
        if isinstance(x, float) and not x.is_integer():
            raise ValueError(
                "The ACDCPruningModifier assumes that attributes"
                "`start_epoch`, `end epoch` and `update frequency` "
                "are integers, or floats which evaluate to integers."
                "However: type(x)==float and x.is_integer() == False."
            )
        return int(x)
