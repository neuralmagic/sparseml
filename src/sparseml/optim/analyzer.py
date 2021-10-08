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
Code for describing layers / operators in ML framework neural networks.
"""

import json
from typing import Any, Dict, List, Tuple

from sparseml.utils import clean_path, create_parent_dirs


__all__ = ["AnalyzedLayerDesc"]


class AnalyzedLayerDesc(object):
    """
    Description of an executed neural network layer.
    Contains information about the number of flops, shapes, params, etc.

    :param name: name of the layer
    :param type_: type of the layer
    :param params: number of parameters of the layer
    :param zeroed_params: number of parameters with values of zero
    :param prunable_params: number of parameters that could be pruned
    :param params_dims: dimensions of parameters
    :param prunable_params_dims: dimensions of prunable parameters
    :param execution_order: execution order of the layer/operation
    :param input_shape: shapes of input tensors
    :param output_shape: shapes of output tensors
    :param flops: Unused
    :param total_flops: total number of float operations
    """

    @staticmethod
    def save_descs(descs: List, path: str):
        """
        Save a list of AnalyzedLayerDesc to a json file

        :param descs: a list of descriptions to save
        :param path: the path to save the descriptions at
        """
        path = clean_path(path)
        create_parent_dirs(path)
        save_obj = {"descriptions": [desc.dict() for desc in descs]}

        with open(path, "w") as file:
            json.dump(save_obj, file)

    @staticmethod
    def load_descs(path: str) -> List:
        """
        Load a list of AnalyzedLayerDesc from a json file

        :param path: the path to load the descriptions from
        :return: the loaded list of AnalyzedLayerDesc
        """
        path = clean_path(path)

        with open(path, "r") as file:
            obj = json.load(file)

        descs = []

        for desc_obj in obj["descriptions"]:
            desc_obj["type_"] = desc_obj["type"]
            del desc_obj["type"]
            del desc_obj["terminal"]
            del desc_obj["prunable"]
            descs.append(AnalyzedLayerDesc(**desc_obj))

        return descs

    @staticmethod
    def merge_descs(orig, descs: List):
        """
        Merge a layer description with a list of others

        :param orig: original description
        :param descs: list of descriptions to merge with
        :return: a combined description
        """
        merged = AnalyzedLayerDesc(
            name=orig.name,
            type_=orig.type_,
            params=orig.params,
            zeroed_params=orig.zeroed_params,
            prunable_params=orig.prunable_params,
            params_dims=orig.params_dims,
            prunable_params_dims=orig.prunable_params_dims,
            execution_order=orig.execution_order,
            input_shape=orig.input_shape,
            output_shape=orig.output_shape,
            flops=orig.flops,
            total_flops=orig.total_flops,
            stride=orig.stride,
        )

        for desc in descs:
            merged.flops += desc.flops
            merged.total_flops += desc.total_flops
            merged.params += desc.params
            merged.prunable_params += desc.prunable_params
            merged.zeroed_params += desc.zeroed_params

        return merged

    def __init__(
        self,
        name: str,
        type_: str,
        params: int = 0,
        zeroed_params: int = 0,
        prunable_params: int = 0,
        params_dims: Dict[str, Tuple[int, ...]] = None,
        prunable_params_dims: Dict[str, Tuple[int, ...]] = None,
        execution_order: int = -1,
        input_shape: Tuple[Tuple[int, ...], ...] = None,
        output_shape: Tuple[Tuple[int, ...], ...] = None,
        flops: int = 0,
        total_flops: int = 0,
        stride: Tuple[int, ...] = None,
    ):
        self.name = name
        self.type_ = type_

        self.params = params
        self.prunable_params = prunable_params
        self.zeroed_params = zeroed_params
        self.params_dims = params_dims
        self.prunable_params_dims = prunable_params_dims

        self.execution_order = execution_order
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.flops = flops
        self.total_flops = total_flops
        self.stride = stride

    def __repr__(self):
        return "AnalyzedLayerDesc({})".format(self.dict())

    @property
    def terminal(self) -> bool:
        """
        :return: True if this is a terminal op, ie it is doing compute and is not
            a container, False otherwise
        """
        return self.params_dims is not None

    @property
    def prunable(self) -> bool:
        """
        :return: True if the layer supports kernel sparsity (is prunable),
            False otherwise
        """
        return self.prunable_params > 0

    def dict(self) -> Dict[str, Any]:
        """
        :return: A serializable dictionary representation of the current instance
        """

        return {
            "name": self.name,
            "type": self.type_,
            "params": self.params,
            "zeroed_params": self.zeroed_params,
            "prunable_params": self.prunable_params,
            "params_dims": self.params_dims,
            "prunable_params_dims": self.prunable_params_dims,
            "execution_order": self.execution_order,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "stride": self.stride,
            "flops": self.flops,
            "total_flops": self.total_flops,
            "terminal": self.terminal,
            "prunable": self.prunable,
        }
