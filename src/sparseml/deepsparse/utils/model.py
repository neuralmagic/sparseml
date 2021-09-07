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
Utilities for ONNX models and running inference with them in DeepSparse
"""
import logging
import tempfile
import time
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy
import psutil
from onnx import ModelProto

from sparseml.onnx.utils import DataLoader, ModelRunner
from sparseml.onnx.utils.helpers import (
    check_load_model,
    extract_node_id,
    get_node_by_id,
    get_prunable_node_from_foldable,
    is_foldable_node,
)
from sparsezoo.objects import File, Model


try:
    import deepsparse
    from deepsparse import Scheduler, analyze_model, compile_model
    from deepsparse.cpu import cpu_details
except Exception:
    deepsparse = None
    compile_model = None
    analyze_model = None
    cpu_details = None
    Scheduler = None

__all__ = [
    "max_available_cores",
    "DeepSparseModelRunner",
    "DeepSparseAnalyzeModelRunner",
]

_LOGGER = logging.getLogger(__name__)


def _check_args(args, kwargs):
    if args:
        raise ValueError(
            "args was not empty, cannot pass any additional args through: {}".format(
                args
            )
        )

    if kwargs:
        raise ValueError(
            (
                "kwargs was not empty, cannot pass any additional args through: {}"
            ).format(kwargs)
        )


def max_available_cores() -> int:
    """
    :return: the maximum number of physical cores detected on the system
    """
    if cpu_details is not None:
        _LOGGER.debug(
            "retrieving physical core count per socket "
            "from deepsparse.cpu.cpu_details()"
        )

        return cpu_details()[0]

    _LOGGER.debug("retrieving physical core count using psutil")
    physical_cores = psutil.cpu_count(logical=False)

    return physical_cores if physical_cores else -1


class _DeepSparseBaseModelRunner(ModelRunner):
    @staticmethod
    def available() -> bool:
        """
        :return: True if deepsparse package is available, False otherwise
        """
        return compile_model is not None

    def __init__(
        self,
        model: Union[str, ModelProto, Model, File],
        batch_size: int,
        num_cores: int,
        loss: Union[
            Callable[[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]], Any], None
        ],
    ):
        if not _DeepSparseBaseModelRunner.available():
            raise ModuleNotFoundError(
                "deepsparse is not installed on the system, "
                "must be installed before using any ModelRunner for deepsparse"
            )

        super().__init__(loss)
        self._model = model

        self._batch_size = batch_size
        self._num_cores = num_cores

        if not (
            isinstance(self._model, str)
            or isinstance(self._model, File)
            or isinstance(self._model, Model)
        ):
            self._model_tmp = tempfile.NamedTemporaryFile(delete=True)
            self._model_tmp.write(self._model.SerializeToString())
            self._model_tmp.flush()
            self._model = self._model_tmp.name
        else:
            self._model_tmp = None

    def __del__(self):
        try:
            if self._model_tmp is not None:
                self._model_tmp.close()

            self._model_tmp = None
        except Exception:
            pass

    @abstractmethod
    def batch_forward(
        self, batch: Dict[str, numpy.ndarray], *args, **kwargs
    ) -> Tuple[Any, float]:
        raise NotImplementedError()


class DeepSparseModelRunner(_DeepSparseBaseModelRunner):
    """
    Class for handling running inference for an ONNX model through Neural Magic

    :param model: the path to the ONNX model file or the loaded onnx.ModelProto
    :param batch_size: the size of the batch to create the model for
    :param num_cores: The number of physical cores to run the model on.
        Pass None or 0 to run on the max number of cores
        in one socket for the current machine, default None
    :param num_sockets: The number of physical sockets to run the model on.
        Pass None or 0 to run on the max number of sockets for the
        current machine, default None
    :param scheduler: The kind of scheduler to execute with. Pass None for the default.
    :param loss: the loss function, if any, to run for evaluation of the model
    """

    def __init__(
        self,
        model: Union[str, ModelProto, Model, File],
        batch_size: int,
        num_cores: int = None,
        num_sockets: int = None,
        scheduler: Scheduler = None,
        loss: Union[
            Callable[[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]], Any], None
        ] = None,
    ):
        super().__init__(model, batch_size, num_cores, loss)
        self._engine = compile_model(
            self._model,
            batch_size=batch_size,
            num_cores=num_cores,
            num_sockets=num_sockets,
            scheduler=scheduler,
        )
        _LOGGER.debug("created model in neural magic {}".format(self._engine))

    def __del__(self):
        super().__del__()

        try:
            del self._engine
        except Exception:
            pass

    def __repr__(self):
        return str(self._engine)

    def run(
        self,
        data_loader: "DataLoader",
        desc: str = "",
        show_progress: bool = True,
        max_steps: int = -1,
        *args,
        **kwargs,
    ) -> Tuple[List[Any], List[float]]:
        """
        Run inference for a model for the data given in the data_loader iterator
        through neural magic inference engine.

        :param data_loader: the data_loader used to load batches of data to
            run through the model
        :param desc: str to display if show_progress is True
        :param show_progress: True to show a tqdm bar when running, False otherwise
        :param max_steps: maximum number of steps to take for the data_loader
            instead of running over all the data
        :return: a tuple containing the list of outputs and the list of times
            for running the data
        """
        _check_args(args, kwargs)

        return super().run(data_loader, desc, show_progress, max_steps)

    def batch_forward(
        self, batch: Dict[str, numpy.ndarray], *args, **kwargs
    ) -> Tuple[Dict[str, numpy.ndarray], float]:
        """
        :param batch: the batch to run through the ONNX model for inference
            in the DeepSparse Engine
        :return: a tuple containing the result of the inference,
            the time to perform the inference
        """
        _check_args(args, kwargs)
        nm_batch = list(batch.values())
        pred_time = time.time()
        pred = self._engine.mapped_run(nm_batch)
        pred_time = time.time() - pred_time

        return pred, pred_time


class DeepSparseAnalyzeModelRunner(_DeepSparseBaseModelRunner):
    """
    Class for handling running inference for an ONNX model through Neural Magic's
    analyze_model api

    :param model: the path to the ONNX model file or the loaded onnx.ModelProto
    :param batch_size: the size of the batch to create the model for
    :param num_cores: the number of physical cores to run the model on. Defaults
        to run on all available cores
    """

    def __init__(
        self,
        model: Union[str, ModelProto],
        batch_size: int,
        num_cores: int = None,
    ):
        super().__init__(model, batch_size, num_cores, loss=None)

    def run(
        self,
        data_loader: "DataLoader",
        desc: str = "",
        show_progress: bool = True,
        max_steps: int = 1,
        num_iterations: int = 20,
        num_warmup_iterations: int = 5,
        optimization_level: int = 1,
        imposed_ks: Union[None, float] = None,
        *args,
        **kwargs,
    ) -> Tuple[List[Dict], List[float]]:
        """
        Run inference for a model for the data given in the data_loader iterator
        through neural magic inference engine model analysis function.
        The analysis function allows more granular control over how the
        model is executed such as optimization levels and imposing kernel sparsity.
        In addition, gives back layer by layer timings that were run through.

        :param data_loader: the data_loader used to load batches of data to
            run through the model
        :param desc: str to display if show_progress is True
        :param show_progress: True to show a tqdm bar when running, False otherwise
        :param max_steps: maximum number of steps to take for the data_loader
            instead of running over all the data
        :param num_iterations: number of iterations to run the analysis benchmark for
        :param num_warmup_iterations: number of iterations to run warmup for
            before benchmarking
        :param optimization_level: the optimization level to use in neural magic;
            1 for optimizations on, 0 for limited optimizations
        :param imposed_ks: kernel sparsity value to impose on all the prunable
            layers in the model. None or no imposed sparsity
        :return: a tuple containing the performance results for the run as returned
            from the analyze_model function, total time to run them
        """
        _check_args(args, kwargs)

        return super().run(
            data_loader,
            desc,
            show_progress,
            max_steps,
            num_iterations=num_iterations,
            num_warmup_iterations=num_warmup_iterations,
            optimization_level=optimization_level,
            imposed_ks=imposed_ks,
        )

    def batch_forward(
        self,
        batch: Dict[str, numpy.ndarray],
        num_iterations: int = 1,
        num_warmup_iterations: int = 0,
        optimization_level: int = 1,
        imposed_ks: Union[None, float] = None,
        *args,
        **kwargs,
    ) -> Tuple[Dict[str, numpy.ndarray], float]:
        """
        :param batch: the batch to run through the ONNX model for inference
            benchmarking analysis in the neural magic system
        :param num_iterations: number of iterations to run the analysis benchmark for
        :param num_warmup_iterations: number of iterations to run warmup for
            before benchmarking
        :param optimization_level: the optimization level to use in neural magic;
            1 for optimizations on, 0 for limited optimizations
        :param imposed_ks: kernel sparsity value to impose on all the prunable
            layers in the model. None or no imposed sparsity
        :return: a tuple containing the result of the inference,
            the time to perform the inference
        """
        _check_args(args, kwargs)
        nm_batch = list(batch.values())
        pred_time = time.time()
        nm_pred = analyze_model(
            self._model,
            nm_batch,
            self._batch_size,
            self._num_cores,
            num_iterations,
            num_warmup_iterations,
            optimization_level,
            imposed_ks=imposed_ks,
        )
        pred_time = time.time() - pred_time
        split_canonical_names(nm_pred)
        correct_nm_analyze_model_node_ids(nm_pred, self._model)

        return nm_pred, pred_time


def correct_nm_analyze_model_node_ids(nm_result: Dict, model: Union[str, ModelProto]):
    """
    Correct the node ids returned from the deepsparse.analyze_model api.
    In some cases, it will return the ids for folded nodes due to ONNXRuntime folding.
    This finds the corrected node ids from those folded nodes.
    Additionally, ops that did not have an id are changed from the returned
    string <none> to proper None python type

    :param nm_result: the result from the deepsparse.analyze_model api
    :param model: the onnx model proto or path to the onnx file that the
        nm_result was for
    """
    model = check_load_model(model)

    for layer in nm_result["layer_info"]:
        node_id = (
            layer["canonical_name"] if "<none>" not in layer["canonical_name"] else None
        )

        if node_id is None:
            layer["canonical_name"] = None
            continue

        node = get_node_by_id(model, node_id)

        if node is None:
            _LOGGER.warning(
                (
                    "node returned from deepsparse.analyze_model "
                    "was not found in the model graph; node id {}"
                ).format(node_id)
            )
            continue

        if is_foldable_node(node):
            _LOGGER.debug(
                "foldable node of id {} returned from "
                "deepsparse.analyze_model api, matching to prunable node"
            )
            # traverse previous because incorrect node id will only be returned
            # for following foldable layers, not previous
            node = get_prunable_node_from_foldable(model, node, traverse_previous=True)

            if node is None:
                _LOGGER.warning(
                    (
                        "could not find prunable node from a foldable node "
                        "returned in the deepsparse.analyze_model api; "
                        "node id: {}"
                    ).format(node_id)
                )
            else:
                prunable_node_id = extract_node_id(node)
                _LOGGER.debug(
                    (
                        "matched prunable node of id {} to foldable node {} as "
                        "returned from deepsparse.analyze_model api"
                    ).format(prunable_node_id, node_id)
                )
                layer["canonical_name"] = prunable_node_id


def split_canonical_names(nm_result: Dict):
    """
    Splits analysis layer results from grouped canonical names by individual nodes.
    Stores the original grouped canonical name in the 'meta_canonical_name' field.

    Will split on any canonical_name that includes ','.

    :param nm_result: the result from the deepsparse.analyze_model api
    """
    split_layer_infos = []
    for layer in nm_result["layer_info"]:
        if "," in layer["canonical_name"]:
            for sub_layer_name in layer["canonical_name"].split(","):
                sub_layer_info = deepcopy(layer)
                sub_layer_info["meta_canonical_name"] = layer["canonical_name"]
                sub_layer_info["canonical_name"] = sub_layer_name
                split_layer_infos.append(sub_layer_info)
        else:
            layer["meta_canonical_name"] = None
            split_layer_infos.append(layer)
    nm_result["layer_info"] = split_layer_infos
