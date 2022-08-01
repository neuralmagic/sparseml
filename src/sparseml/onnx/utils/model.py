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
Utilities for ONNX models and running inference with them
"""
import logging
import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy
import psutil
from onnx import ModelProto
from tqdm import auto

from sparseml.onnx.base import require_onnxruntime
from sparseml.onnx.utils.data import DataLoader
from sparseml.onnx.utils.graph_editor import override_model_batch_size
from sparseml.onnx.utils.helpers import (
    check_load_model,
    extract_node_id,
    get_node_by_id,
    get_prunable_node_from_foldable,
    is_foldable_node,
)
from sparsezoo import File, Model


try:
    import deepsparse
    from deepsparse import analyze_model, compile_model
    from deepsparse.cpu import cpu_details
except Exception:
    deepsparse = None
    compile_model = None
    analyze_model = None
    cpu_details = None


try:
    from openvino.inference_engine import IECore, IENetwork, StatusCode, get_version
except Exception:
    IENetwork, IECore, get_version, StatusCode = None, None, None, None


__all__ = [
    "max_available_cores",
    "ModelRunner",
    "ORTModelRunner",
    "DeepSparseModelRunner",
    "OpenVINOModelRunner",
    "correct_nm_analyze_model_node_ids",
    "split_canonical_names",
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


class ModelRunner(ABC):
    """
    Abstract class for handling running inference for a model

    :param loss: the loss function, if any, to run for evaluation of the model
    """

    def __init__(
        self,
        loss: Union[
            Callable[[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]], Any], None
        ] = None,
    ):
        self._loss = loss

    def run(
        self,
        data_loader: DataLoader,
        desc: str = "",
        show_progress: bool = True,
        max_steps: int = -1,
        *args,
        **kwargs,
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
        for output, pred_time in self.run_iter(
            data_loader, desc, show_progress, max_steps, *args, **kwargs
        ):
            outputs.append(output)
            times.append(pred_time)

        return outputs, times

    def run_iter(
        self,
        data_loader: DataLoader,
        desc: str = "",
        show_progress: bool = True,
        max_steps: int = -1,
        *args,
        **kwargs,
    ):
        """
        Iteratively runs inference for a model for the data given in the
        data_loader iterator

        :param data_loader: the data_loader used to load batches of data to
            run through the model
        :param desc: str to display if show_progress is True
        :param show_progress: True to show a tqdm bar when running, False otherwise
        :param max_steps: maximum number of steps to take for the data_loader
            instead of running over all the data
        :return: an iterator to go through the tuples containing
            the list of outputs and the list of times for running the data
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


class OpenVINOModelRunner(ModelRunner):
    """
    OpenVINO model runner class

    :param model: The path to the IR xml file after conversion
    :param loss: loss function to run evaluation
    :param nthreads: number of threads to run the model
    :param batch_size: Batch size value. If not specified, the batch size value is
        determined from Intermediate Representation
    :param shape: shape to be set for the input(s).
        For example, "input1[1,3,224,224],input2[1,4]"
        or "[1,3,224,224]" in case of one input size.
    """

    @staticmethod
    def available() -> bool:
        return IECore is not None

    def __init__(
        self,
        model: str,  # TODO: accept ONNX as input
        loss: Union[
            Callable[[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]], Any], None
        ] = None,
        nthreads: int = 1,
        batch_size: int = 0,
        shape: str = "",
    ):
        super().__init__(loss)
        self._loss = loss

        self._nthreads = nthreads
        self._batch_size = batch_size
        self._shape = shape

        # Loading inference engine
        self._ie = IECore()

        # Setting device configuration (for device 'CPU')
        cpu_threads_num = str(self._nthreads)
        cpu_bind_thread = "YES"
        config = {
            "CPU_THREADS_NUM": cpu_threads_num,
            "CPU_BIND_THREAD": cpu_bind_thread,
        }
        self._ie.set_config(config, "CPU")

        # Read the Intermediate Representation of the network
        self._ie_network = self._read_network(model)

        # Resizing network to match image sizes and given batch
        self._resize_network()

        # Configuring input of the model
        self._config_network_inputs()

        # Load the model to the device
        self._exe_network = self._load_network()

    def batch_forward(
        self, batch: Dict[str, numpy.ndarray], *args, **kwargs
    ) -> Tuple[Any, float]:
        """
        Run a batch through the model

        :param batch: batch of data
        :return: result of the inference as dictionary, and the inference time
        """
        self._set_inputs(batch)
        infer_request = self._exe_network.requests[0]

        pred_time = time.time()
        infer_request.infer()
        pred_time = time.time() - pred_time

        return infer_request.output_blobs, pred_time

    def network_input_shapes(self):
        """
        Get network input shapes
        :return: dictionary of shapes for each input key
        """
        assert self._ie_network is not None
        input_shapes = {}
        for k, v in self._ie_network.input_info.items():
            shape_with_batch = v.input_data.shape.copy()
            input_shapes[k] = shape_with_batch[1:]
        return input_shapes

    def _read_network(self, model_file_path: str):
        """
        Read the IR into IE network
        :param model_file_path: path to the IR file
        """
        model_filename = os.path.abspath(model_file_path)
        head, ext = os.path.splitext(model_filename)
        weights_filename = os.path.abspath(head + ".bin") if ext == ".xml" else ""
        ie_network = self._ie.read_network(model_filename, weights_filename)
        return ie_network

    def _update_shapes(self, shapes, shapes_string: str, inputs_info):
        """
        Check if the network input shapes need update

        :param shape: dictionary of input shapes; updated to new shapes if needed
        :param shapes_string: string of shapes from user inputs
        :param inputs_info: inputs of the network read from the IR file

        :return: if the network input shapes need updated
        """
        updated = False
        matches = re.findall(r"(.*?)\[(.*?)\],?", shapes_string)
        if matches:
            for match in matches:
                input_name = match[0]
                parsed_shape = [int(dim) for dim in match[1].split(",")]
                if input_name != "":
                    shapes[input_name] = parsed_shape
                    updated = True
                else:
                    shapes.update({k: parsed_shape for k in shapes.keys()})
                    updated = True
                    break
        else:
            raise Exception("Can't parse `shape` parameter: {}".format(shapes_string))
        return updated

    def _adjust_shapes_batch(self, shapes, batch_size: int, inputs_info):
        """
        Check and change the shape of network input to fit the batch size
        :param shapes: dictionary of input shapes; updated if needed
        :param batch_size: batch size
        :param inputs_info: inputs of the network read from the IR file
        :return: if the network input shapes need updated
        """
        updated = False
        for name, data in inputs_info.items():
            layout = data.input_data.layout
            batch_index = layout.index("N") if "N" in layout else -1
            if batch_index != -1 and shapes[name][batch_index] != batch_size:
                shapes[name][batch_index] = batch_size
                updated = True
        return updated

    def _resize_network(self):
        """
        Resize the network input shapes
        """
        shapes = {
            k: v.input_data.shape.copy() for k, v in self._ie_network.input_info.items()
        }
        reshape = False
        if self._shape:
            reshape |= self._update_shapes(
                shapes, self._shape, self._ie_network.input_info
            )
        if self._batch_size and self._batch_size != self._ie_network.batch_size:
            reshape |= self._adjust_shapes_batch(
                shapes, self._batch_size, self._ie_network.input_info
            )

        if reshape:
            _LOGGER.info(
                "Reshaping network: {}".format(
                    ", ".join("'{}': {}".format(k, v) for k, v in shapes.items())
                )
            )
            self._ie_network.reshape(shapes)

    def _is_image(self, blob):
        """
        Check if the input data is image
        """
        if blob.layout != "NCHW":
            return False
        channels = blob.shape[1]
        return channels == 3

    def _config_network_inputs(self):
        """
        Configure network inputs
        """
        input_info = self._ie_network.input_info
        for key in input_info.keys():
            if self._is_image(input_info[key].input_data):
                # Set the precision of input data provided by the user
                # Should be called before load of the network to the plugin
                input_info[key].precision = "U8"

    def _load_network(self):
        """
        Load the network with given device and request info
        """
        exe_network = self._ie.load_network(
            self._ie_network, "CPU", config={}, num_requests=1
        )
        return exe_network

    def _set_inputs(self, batch: Dict[str, numpy.ndarray]):
        """
        Set the inputs for the infer request object

        :param batch: batch of data
        """
        infer_requests = self._exe_network.requests
        assert len(infer_requests) == 1
        inputs = infer_requests[0].input_blobs
        for k, v in batch.items():
            if k not in inputs.keys():
                raise Exception("No input with name {} found!".format(k))
            inputs[k].buffer[:] = v


class ORTModelRunner(ModelRunner):
    """
    Class for handling running inference for an ONNX model through onnxruntime

    :param model: the path to the ONNX model file or the loaded onnx.ModelProto
    :param loss: the loss function, if any, to run for evaluation of the model
    :param overwrite_input_names: True to overwrite the input data names to what
        is found in for the model inputs, False to keep as found in the loaded data
    :param nthreads: number of threads used to run the model (single node);
        default to 0 for onnxruntime to choose
    :param batch_size: if provided, and the model has a hardcoded batch size, will
        rewrite the model proto so that the model batch size matches batch_size
    :param providers: list of ORT provider names. will default to
        ort.get_available_providers()
    """

    @require_onnxruntime()
    def __init__(
        self,
        model: Union[str, ModelProto],
        loss: Union[
            Callable[[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]], Any], None
        ] = None,
        overwrite_input_names: bool = True,
        nthreads: int = 0,
        batch_size: int = None,
        providers: List[str] = None,
        **kwargs,
    ):
        import onnxruntime  # import protected by @require_onnxruntime()

        super().__init__(loss)
        self._model = check_load_model(model)

        if batch_size is not None:
            override_model_batch_size(self._model, batch_size)
        # default selected providers to ort default
        providers = providers or onnxruntime.get_available_providers()

        sess_options = onnxruntime.SessionOptions()

        # Note: If ORT was built with OpenMP, use OpenMP env variable such as
        # OMP_NUM_THREADS to control the number of threads.
        # See: https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_Perf_Tuning.md  # noqa
        sess_options.intra_op_num_threads = nthreads

        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self._session = onnxruntime.InferenceSession(
            self._model.SerializeToString()
            if not isinstance(self._model, str)
            else self._model,
            sess_options,
            providers=providers,
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
        **kwargs,
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
    :param num_cores: the number of physical cores to run the model on. Defaults
        to run on all available cores
    :param loss: the loss function, if any, to run for evaluation of the model
    """

    def __init__(
        self,
        model: Union[str, ModelProto],
        batch_size: int,
        num_cores: int = None,
        loss: Union[
            Callable[[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]], Any], None
        ] = None,
    ):
        super().__init__(model, batch_size, num_cores, loss)
        self._engine = compile_model(
            self._model, batch_size=batch_size, num_cores=num_cores
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
        data_loader: DataLoader,
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
        data_loader: DataLoader,
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
