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
Helper functions for retrieving information related to model sparsification
"""

import json
from contextlib import nullcontext
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Tuple

import torch
from torch.nn import Module
from tqdm import tqdm

from sparseml.pytorch.utils.helpers import (
    get_prunable_layers,
    get_quantizable_layers,
    get_quantized_layers,
    tensor_sparsity,
)


__all__ = [
    "ModuleSparsificationInfo",
    "GradSampler",
]


class ModuleSparsificationInfo:
    """
    Helper class for providing information related to torch Module parameters
    and the amount of sparsification applied. Includes information for pruning
    and quantization

    :param module: torch Module to analyze
    """

    def __init__(self, module: Module):
        self.module = module
        self.trainable_params = list(
            filter(lambda param: param.requires_grad, self.module.parameters())
        )

    def __str__(self):
        return json.dumps(
            {
                "params_summary": {
                    "total": self.params_total,
                    "sparse": self.params_sparse,
                    "sparsity_percent": self.params_sparse_percent,
                    "prunable": self.params_prunable_total,
                    "prunable_sparse": self.params_prunable_sparse,
                    "prunable_sparsity_percent": self.params_prunable_sparse_percent,
                    "quantizable": self.params_quantizable,
                    "quantized": self.params_quantized,
                    "quantized_percent": self.params_quantized_percent,
                },
                "params_info": self.params_info,
            }
        )

    @property
    def params_total(self) -> int:
        """
        :return: total number of trainable parameters in the model
        """
        return sum(torch.numel(param) for param in self.trainable_params)

    @property
    def params_sparse(self) -> int:
        """
        :return: total number of sparse (0) trainable parameters in the model
        """
        return sum(
            round(tensor_sparsity(param).item() * torch.numel(param))
            for param in self.trainable_params
        )

    @property
    def params_sparse_percent(self) -> float:
        """
        :return: percent of sparsified parameters in the entire model
        """
        return self.params_sparse / float(self.params_total) * 100

    @property
    def params_prunable_total(self) -> int:
        """
        :return: total number of parameters across prunable layers
        """
        return sum(
            torch.numel(layer.weight)
            for (name, layer) in get_prunable_layers(self.module)
        )

    @property
    def params_prunable_sparse(self) -> int:
        """
        :return: total number of sparse (0) parameters across prunable lauyers
        """
        return sum(
            round(tensor_sparsity(layer.weight).item() * torch.numel(layer.weight))
            for (name, layer) in get_prunable_layers(self.module)
        )

    @property
    def params_prunable_sparse_percent(self) -> float:
        """
        :return: percent of prunable parameters that have been pruned
        """
        return self.params_prunable_sparse / float(self.params_prunable_total) * 100

    @property
    def params_quantizable(self) -> int:
        """
        :return: number of parameters that are included in quantizable layers
        """
        return sum(
            torch.numel(layer.weight)
            + (
                torch.numel(layer.bias)
                if hasattr(layer, "bias") and layer.bias is not None
                else 0
            )
            for (name, layer) in get_quantizable_layers(self.module)
        )

    @property
    def params_quantized(self) -> int:
        """
        :return: number of parameters across quantized layers
        """
        return sum(
            torch.numel(layer.weight)
            + (
                torch.numel(layer.bias)
                if hasattr(layer, "bias") and layer.bias is not None
                else 0
            )
            for (name, layer) in get_quantized_layers(self.module)
        )

    @property
    def params_quantized_percent(self) -> float:
        """
        :return: percentage of parameters that have been quantized
        """
        return self.params_quantized / float(self.params_quantizable) * 100

    @property
    def params_info(self) -> Dict[str, Dict]:
        """
        :return: dict of parameter name to its sparsification information
        """
        return {
            f"{name}.weight": {
                "numel": torch.numel(layer.weight),
                "sparsity": tensor_sparsity(layer.weight).item(),
                "quantized": hasattr(layer, "weight_fake_quant"),
            }
            for (name, layer) in get_prunable_layers(self.module)
        }


class GradSampler:
    """
    Class for computing gradient samples for a Model given a sample data loader and
    loss function.

    :param data_loader: iterator of data samples to use as model inputs and their loss
        targets. items must be tuples of
        (forward_args: List, forward_kwargs: Dict, loss_targets: Any)
        where the forward pass will be outputs = model(*forward_args, **forward_kwargs)
        and loss will be loss = loss_fn(outputs, loss_targets)
    :param loss_fn: function to be called on model outputs to compute the loss at
        each step
    """

    def __init__(
        self,
        data_loader: Iterator[Tuple[List[Any], Dict[str, Any], Any]],
        loss_fn: Callable[[Any], Any],
    ):
        if not isinstance(data_loader, Iterable):
            raise ValueError(
                "data_loader for GradSampler must be Iterable, received object of "
                f"type {type(data_loader)}"
            )
        if not callable(loss_fn):
            raise ValueError(
                "loss_fn for GradSampler must be callable, given input "
                f"with type {type(loss_fn)}"
            )

        self._data_loader = data_loader
        self._loss_fn = loss_fn

    def iter_module_backwards(
        self,
        module: Module,
        num_grads: int,
        progress_bar: bool = True,
    ) -> Generator[int, None, None]:
        """
        :param module: module to compute gradients for
        :param num_grads: number of gradient samples to compute
        :return: generator that yields after every gradient is computed with the index
            of the gradient sample number
        """
        computed_grads = 0
        # if progress bar is turned off, use nullcontext which has no effect on loop
        context = (
            tqdm(total=num_grads, desc="Collecting gradients", disable = not progress_bar)
        )

        with context as pbar:
            while computed_grads < num_grads:
                for forward_args, forward_kwargs, loss_target in self._data_loader:
                    module.zero_grad()
                    # run sample forward and backwards pass
                    model_outputs = module(*forward_args, **forward_kwargs)
                    loss = self._loss_fn(model_outputs, loss_target)
                    loss.backward()

                    # yield so gradients can be collected
                    computed_grads += 1
                    yield computed_grads
                    if progress_bar:
                        pbar.update(1)
                    if computed_grads >= num_grads:
                        break
        module.zero_grad()
