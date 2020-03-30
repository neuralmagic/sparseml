import json
from typing import Any, Dict, Tuple

__all__ = ["AnalyzedLayerDesc"]


class AnalyzedLayerDesc(object):
    """
    Description of a module's layer
    """

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
    ):
        """
        :param name: name of the layer
        :param type_: type of the layer
        :param params: number of parameters of the layer
        :param zeroed_params: number of parameters with values of zero
        :param prunable_params: number of parameters that could be pruned
        :param params_dims: dimensions of parameters
        :param prunable_params_dims: dimensions of prunable parameters
        :param execution_order: execution order of the layer/operation
        :param input_shape: shapes of input tensors
        :param output_shape: shaps of output tensors
        :param flops: Unused
        :param total_flops: total number of float operations
        """
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

    def __repr__(self):
        """ String representation of the instance """
        return "AnalyzedLayerDesc({})".format(self.json())

    @property
    def terminal(self) -> bool:
        return self.params_dims is not None

    @property
    def prunable(self) -> bool:
        """ Whether the layer could be pruned """
        return self.prunable_params > 0

    def dict(self) -> Dict[str, Any]:
        """ Export to dictionary data structure """
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
            "flops": self.flops,
            "total_flops": self.total_flops,
            "terminal": self.terminal,
            "prunable": self.prunable,
        }

    def json(self) -> str:
        """ Save to json format """
        return json.dumps(self.dict())

    @staticmethod
    def merge_descs(orig, descs):
        """
        Merge a layer description with a list of others
        :param orig: original description
        :param descs: list of descriptions to merge with
        :return A combined description
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
        )

        for desc in descs:
            merged.flops += desc.flops
            merged.total_flops += desc.total_flops
            merged.params += desc.params
            merged.prunable_params += desc.prunable_params
            merged.zeroed_params += desc.zeroed_params

        return merged
