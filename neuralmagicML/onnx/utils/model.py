"""
Utilities for ONNX models and running inference with them
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Union, Any, Dict, Tuple, List
from collections import OrderedDict
import logging
import tempfile
import time
import psutil
from tqdm import auto
import numpy
from onnx import ModelProto
import onnxruntime

from neuralmagicML.onnx.utils.helpers import (
    check_load_model,
    get_node_by_id,
    is_foldable_node,
    get_prunable_node_from_foldable,
    extract_node_id,
)
from neuralmagicML.onnx.utils.graph_editor import override_model_batch_size
from neuralmagicML.onnx.utils.data import DataLoader

try:
    import neuralmagic
    from neuralmagic import create_model
    from neuralmagic import benchmark_model
    from neuralmagic.cpu import cpu_details
except Exception:
    neuralmagic = None
    create_model = None
    benchmark_model = None
    cpu_details = None

__all__ = [
    "max_available_cores",
    "ModelRunner",
    "ORTModelRunner",
    "NMModelRunner",
    "correct_nm_benchmark_model_node_ids",
    "split_canonical_names",
    "NMBenchmarkModelRunner",
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
            "from neuralmagic.cpu.cpu_details()"
        )

        return cpu_details()[0]

    _LOGGER.debug("retrieving physical core count using psutil")
    physical_cores = psutil.cpu_count(logical=False)

    return physical_cores if physical_cores else -1


class ModelRunner(ABC):
    """
    Abstract class for handling running inference for an ONNX model

    :param model: the path to the ONNX model file or the loaded onnx ModelProto
    :param loss: the loss function, if any, to run for evaluation of the model
    """

    def __init__(
        self,
        model: Union[str, ModelProto],
        loss: Union[
            Callable[[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]], Any], None
        ] = None,
    ):
        self._model = check_load_model(model)
        self._loss = loss

    def run(
        self,
        data_loader: DataLoader,
        desc: str = "",
        show_progress: bool = True,
        max_steps: int = -1,
        *args,
        **kwargs
    ) -> Tuple[List[Any], List[float]]:
        """
        Run inference for a model for the data given in the data_loader iterator

        :param data_loader: the data_loader used to load batches of data to
            run through the model
        :param desc: str to display if show_progress is True
        :param show_progress: True to show a tqdm bar when running, False otherwise
        :param max_steps: maximum number of steps to take for the data_loader
            instead of running over all the data
        :return: a tuple containing the list of outputs and the list of times
            for running the data
        """
        outputs = []
        times = []
        for output, time in self.run_iter(data_loader, desc, show_progress, max_steps, *args, **kwargs):
            outputs.append(output)
            times.append(time)

        return outputs, times

    def run_iter(
        self,
        data_loader: DataLoader,
        desc: str = "",
        show_progress: bool = True,
        max_steps: int = -1,
        *args,
        **kwargs
    ):
        """
        Iteratively runs inference for a model for the data given in the data_loader iterator

        :param data_loader: the data_loader used to load batches of data to
            run through the model
        :param desc: str to display if show_progress is True
        :param show_progress: True to show a tqdm bar when running, False otherwise
        :param max_steps: maximum number of steps to take for the data_loader
            instead of running over all the data
        :return: a tuple containing the list of outputs and the list of times
            for running the data
        """
        counter_len = len(data_loader) if not data_loader.infinite else None

        if max_steps > 0 and counter_len is not None and counter_len > 0:
            progress_steps = min(max_steps, counter_len)
        elif max_steps > 0:
            progress_steps = max_steps
        elif counter_len is not None and counter_len > 0:
            progress_steps = counter_len
        else:
            progress_steps = None

        _LOGGER.debug("running {} items through model".format(progress_steps))
        data_iter = (
            enumerate(data_loader)
            if not show_progress
            else auto.tqdm(enumerate(data_loader), desc=desc, total=progress_steps)
        )

        outputs = []
        times = []

        for batch, (data, label) in data_iter:
            _LOGGER.debug("calling batch_forward for batch {}".format(batch))
            pred, pred_time = self.batch_forward(data, *args, **kwargs)
            _LOGGER.debug("prediction completed in {}".format(pred_time))
            output = self._loss(pred, label) if self._loss is not None else pred
            yield output, pred_time
            if batch >= max_steps - 1 and max_steps > -1:
                break

    @abstractmethod
    def batch_forward(
        self, batch: Dict[str, numpy.ndarray], *args, **kwargs
    ) -> Tuple[Any, float]:
        """
        Abstract method for subclasses to override to run a batch through the
        inference engine for the ONNX model it was constructed with

        :param batch: the batch to run through the ONNX model for inference
        :return: a tuple containing the result of the inference,
            the time to perform the inference
        """
        raise NotImplementedError()


class ORTModelRunner(ModelRunner):
    """
    Class for handling running inference for an ONNX model through onnxruntime

    :param model: the path to the ONNX model file or the loaded onnx.ModelProto
    :param loss: the loss function, if any, to run for evaluation of the model
    :param overwrite_input_names: True to overwrite the input data names to what
        is found in for the model inputs, False to keep as found in the loaded data
    :param batch_size: if provided, and the model has a hardcoded batch size, will
        rewrite the model proto so that the model batch size matches batch_size
    """

    def __init__(
        self,
        model: Union[str, ModelProto],
        loss: Union[
            Callable[[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]], Any], None
        ] = None,
        overwrite_input_names: bool = True,
        batch_size: int = None,
    ):
        super().__init__(model, loss)
        if batch_size is not None:
            override_model_batch_size(self._model, batch_size)
        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self._session = onnxruntime.InferenceSession(
            self._model.SerializeToString()
            if not isinstance(self._model, str)
            else self._model,
            sess_options,
        )
        self._overwrite_input_names = overwrite_input_names
        _LOGGER.debug("created model in onnxruntime {}".format(self._session))

    def __del__(self):
        try:
            del self._session
        except Exception:
            pass

    def __repr__(self):
        return str(self._session)

    def run(
        self,
        data_loader: DataLoader,
        desc: str = "",
        show_progress: bool = True,
        max_steps: int = -1,
        *args,
        **kwargs
    ) -> Tuple[List[Any], List[float]]:
        """
        Run inference for a model for the data given in the data_loader iterator
        through ONNX Runtime.

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
            iin onnxruntime
        :return: a tuple containing the result of the inference,
            the time to perform the inference
        """
        _check_args(args, kwargs)

        if not self._overwrite_input_names:
            sess_batch = batch
        else:
            sess_batch = {}
            batch_keys = list(batch.keys())
            _LOGGER.debug(
                "remapping input dict from {} to {}".format(
                    batch_keys, [inp.name for inp in self._session.get_inputs()]
                )
            )

            for inp_index, inp in enumerate(self._session.get_inputs()):
                sess_batch[inp.name] = batch[batch_keys[inp_index]]

        sess_outputs = [out.name for out in self._session.get_outputs()]

        pred_time = time.time()
        pred = self._session.run(sess_outputs, sess_batch)
        pred_time = time.time() - pred_time

        pred_dict = OrderedDict((key, val) for key, val in zip(sess_outputs, pred))

        return pred_dict, pred_time


class _NMModelRunner(ModelRunner):
    @staticmethod
    def available() -> bool:
        """
        :return: True if neuralmagic package is available, False otherwise
        """
        return create_model is not None

    def __init__(
        self,
        model: Union[str, ModelProto],
        batch_size: int,
        num_cores: int,
        loss: Union[
            Callable[[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]], Any], None
        ],
    ):
        if not _NMModelRunner.available():
            raise ModuleNotFoundError(
                "neuralmagic is not installed on the system, "
                "must be installed before using any ModelRunner for neuralmagic"
            )

        super().__init__(model, loss)
        self._batch_size = batch_size
        self._num_cores = num_cores

        if not isinstance(self._model, str):
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


class NMModelRunner(_NMModelRunner):
    """
    Class for handling running inference for an ONNX model through Neural Magic

    :param model: the path to the ONNX model file or the loaded onnx.ModelProto
    :param batch_size: the size of the batch to create the model for
    :param num_cores: the number of physical cores to run the model on
    :param loss: the loss function, if any, to run for evaluation of the model
    """

    def __init__(
        self,
        model: Union[str, ModelProto],
        batch_size: int,
        num_cores: int = -1,
        loss: Union[
            Callable[[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]], Any], None
        ] = None,
    ):
        super().__init__(model, batch_size, num_cores, loss)
        self._nm_model = create_model(
            self._model, batch_size=batch_size, num_cores=num_cores
        )
        _LOGGER.debug("created model in neural magic {}".format(self._nm_model))

    def __del__(self):
        super().__del__()

        try:
            del self._nm_model
        except Exception:
            pass

    def __repr__(self):
        return str(self._nm_model)

    def run(
        self,
        data_loader: DataLoader,
        desc: str = "",
        show_progress: bool = True,
        max_steps: int = -1,
        *args,
        **kwargs
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
            in neuralmagic
        :return: a tuple containing the result of the inference,
            the time to perform the inference
        """
        _check_args(args, kwargs)
        nm_batch = list(batch.values())
        pred_time = time.time()
        pred = self._nm_model.mapped_forward(nm_batch)
        pred_time = time.time() - pred_time

        return pred, pred_time


def correct_nm_benchmark_model_node_ids(nm_result: Dict, model: Union[str, ModelProto]):
    """
    Correct the node ids returned from the neuralmagic.benchmark_model api.
    In some cases, it will return the ids for folded nodes due to ONNXRuntime folding.
    This finds the corrected node ids from those folded nodes.
    Additionally, ops that did not have an id are changed from the returned
    string <none> to proper None python type

    :param nm_result: the result from the neuralmagic.benchmark_model api
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
                    "node returned from neuralmagic.benchmark_model "
                    "was not found in the model graph; node id {}"
                ).format(node_id)
            )
            continue

        if is_foldable_node(node):
            _LOGGER.debug(
                "foldable node of id {} returned from "
                "neuralmagic.benchmark_model api, matching to prunable node"
            )
            # traverse previous because incorrect node id will only be returned
            # for following foldable layers, not previous
            node = get_prunable_node_from_foldable(model, node, traverse_previous=True)

            if node is None:
                _LOGGER.warning(
                    (
                        "could not find prunable node from a foldable node "
                        "returned in the neuralmagic.benchmark_model api; "
                        "node id: {}"
                    ).format(node_id)
                )
            else:
                prunable_node_id = extract_node_id(node)
                _LOGGER.debug(
                    (
                        "matched prunable node of id {} to foldable node {} as "
                        "returned from neuralmagic.benchmark_model api"
                    ).format(prunable_node_id, node_id)
                )
                layer["canonical_name"] = prunable_node_id


def split_canonical_names(nm_result: Dict):
    """
    Splits benchmarking layer results from grouped canonical names by individual nodes.
    Stores the original grouped canonical name in the 'meta_canonical_name' field.

    Will split on any canonical_name that includes ','.

    :param nm_result: the result from the neuralmagic.benchmark_model api
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


class NMBenchmarkModelRunner(_NMModelRunner):
    """
    Class for handling running inference for an ONNX model through Neural Magic's
    benchmark_model api

    :param model: the path to the ONNX model file or the loaded onnx.ModelProto
    :param batch_size: the size of the batch to create the model for
    :param num_cores: the number of physical cores to run the model on
    """

    def __init__(
        self, model: Union[str, ModelProto], batch_size: int, num_cores: int = -1,
    ):
        super().__init__(model, batch_size, num_cores, loss=None)

    def run(
        self,
        data_loader: DataLoader,
        desc: str = "",
        show_progress: bool = True,
        max_steps: int = 1,
        num_iterations: int = 20,
        num_warmup_iterations: int = 5,
        optimization_level: int = 1,
        imposed_ks: Union[None, float] = None,
        *args,
        **kwargs
    ) -> Tuple[List[Dict], List[float]]:
        """
        Run inference for a model for the data given in the data_loader iterator
        through neural magic inference engine benchmarking function.
        The benchmarking function allows more granular control over how the
        model is executed such as optimization levels and imposing kernel sparsity.
        In addition, gives back layer by layer timings that were run through.

        :param data_loader: the data_loader used to load batches of data to
            run through the model
        :param desc: str to display if show_progress is True
        :param show_progress: True to show a tqdm bar when running, False otherwise
        :param max_steps: maximum number of steps to take for the data_loader
            instead of running over all the data
        :param num_iterations: number of iterations to run the benchmark for
        :param num_warmup_iterations: number of iterations to run warmup for
            before benchmarking
        :param optimization_level: the optimization level to use in neural magic;
            1 for optimizations on, 0 for limited optimizations
        :param imposed_ks: kernel sparsity value to impose on all the prunable
            layers in the model. None or no imposed sparsity
        :return: a tuple containing the performance results for the run as returned
            from the benchmark_model function, total time to run them
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
        **kwargs
    ) -> Tuple[Dict[str, numpy.ndarray], float]:
        """
        :param batch: the batch to run through the ONNX model for inference
            benchmarking in the neural magic system
        :param num_iterations: number of iterations to run the benchmark for
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
        nm_pred = benchmark_model(
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
        correct_nm_benchmark_model_node_ids(nm_pred, self._model)

        return nm_pred, pred_time
