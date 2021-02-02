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
Provides a class for performing quantization calibration on an Onnx model.
"""


import os
import tempfile
from typing import Dict, Generator, Iterable, List, Tuple, Union

import numpy as np
import onnx

from sparseml.onnx.utils import ORTModelRunner, fold_conv_bns, get_node_output_nodes


__all__ = ["CalibrationSession"]


class CalibrationSession:
    """
    Class for performing quantization calibration on an Onnx model.

    :param onnx_file: File path to saved Onnx model to calibrate
    :param calibrate_op_types: List of Onnx ops names to calibrate and quantize within
        the model. Currently Onnx only supports quantizing 'Conv' and 'MatMul' ops.
    :param exclude_nodes: List of operator names that should not be quantized
    :param include_nodes: List of operator names to force to be quantized
    :param augmented_model_path: file path to save augmented model to for verification
    :param static: True to use static quantization. Default is True
    """

    def __init__(
        self,
        onnx_file: str,
        calibrate_op_types: Iterable[str] = ("Conv", "MatMul", "Gemm"),
        exclude_nodes: List[str] = None,
        include_nodes: List[str] = None,
        augmented_model_path: str = None,
        static: bool = True,
    ):
        self._onnx_file = onnx_file
        self._calibrate_op_types = list(calibrate_op_types)
        self._exclude_nodes = exclude_nodes or []
        self._include_nodes = include_nodes or []
        self._augmented_model_path = augmented_model_path
        self._static = static

        self._model = onnx.load(self._onnx_file)
        self._optimized_model_path = self._optimize_model()
        self._model_augmented = self.generate_augmented_model()

        if self._augmented_model_path is None:
            self._augmented_model_tmp_file = tempfile.NamedTemporaryFile(
                suffix=".onnx", delete=True
            )
            self._augmented_model_path = self._augmented_model_tmp_file.name
        onnx.save(self._model_augmented, self._augmented_model_path)

        self._sessions = {}  # batch_size -> session
        self._quantization_thresholds = {}  # Dict[node.name, Tuple(min_val, max_val)]

    @property
    def model(self):
        """
        :return: The loaded model, if optimization has run,
            will be the optimized version
        """
        return self._model

    @property
    def model_augmented(self):
        """
        :return: The augmented model, if optimization has run,
            will be the optimized version
        """
        return self._model_augmented

    def _optimize_model(self) -> Union[str, None]:
        """
        Perform batch norm folding in model if possible.
        :return: The tmp file path to the optimized model if optimization is successful
            otherwise returns None and the original model is not changed
        """
        try:
            print("Optimizing {}...".format(self._onnx_file))
            model_optimized = fold_conv_bns(self._onnx_file)
            if model_optimized is None:
                # no optimization performed, skip the rest of this block
                raise Exception()
            onnx.checker.check_model(
                model_optimized
            )  # should raise exception if broken
            optimized_model_path = tempfile.NamedTemporaryFile(
                suffix=".onnx", delete=False
            )
            onnx.save(model_optimized, optimized_model_path.name)
            self._model = model_optimized
            print("Optimization successful")
            return optimized_model_path.name
        except Exception as e:
            print(e)
            print(
                (
                    "WARNING: no conv-batch norms folded for {}, using original model"
                ).format(self._onnx_file)
            )
            return None

    def get_model_input_names(self) -> List[str]:
        """
        :return: List of input names to the model
        """
        return [node.name for node in self._model.graph.input]

    def add_reduce_to_node_output(
        self, node: onnx.NodeProto, output_edge: str, op_type: str
    ) -> Tuple[onnx.NodeProto, onnx.ValueInfoProto]:
        """
        :param node: the node to add the reduce op to
        :param output_edge: the output of node to generate reduce op for
        :param op_type: the reduce operation name
        :return: a tuple of the reduce operation node and its output
        """
        if node is not None and node.name != "":
            reduce_name = node.name + "_{}".format(op_type)
        else:  # Should be an input
            reduce_name = output_edge + "_{}".format(op_type)

        reduce_node = onnx.helper.make_node(
            op_type,
            [output_edge],
            [output_edge + "_{}".format(op_type)],
            reduce_name,
            keepdims=0,
        )

        reduce_node_output = onnx.helper.make_tensor_value_info(
            reduce_node.output[0], onnx.TensorProto.FLOAT, ()
        )
        return reduce_node, reduce_node_output

    def _get_input_node_for_edge(self, input_edge: str) -> onnx.NodeProto:
        """
        :param input_edge: name of graph edge to get input node for
        :return: the node in the original model that is the input to the
            destination of the given input_edge
        """
        for node in self._model.graph.node:
            if input_edge in node.output:
                return node
        return None

    def generate_augmented_model(self) -> onnx.ModelProto:
        """
        return: A new Onnx model with ReduceMin and ReduceMax nodes added to all
            quantizable nodes in the original model and ensures their outputs are
            stored as part of the graph output.
        """

        added_nodes = []
        added_outputs = []
        edges_already_calibrated = []

        for node in self._model.graph.node:
            should_calibrate = (
                (node.op_type in self._calibrate_op_types)
                and (node.name not in self._exclude_nodes)
            ) or (node.name in self._include_nodes)
            if should_calibrate:
                to_calibrate = []

                input_name = node.output[0]
                if input_name not in edges_already_calibrated:
                    edges_already_calibrated.append(input_name)
                    to_calibrate.append((node, input_name))

                if self._static:
                    # In static mode, we precompute the min/max for the inputs as well
                    for input_name in node.input:
                        if input_name not in edges_already_calibrated:
                            edges_already_calibrated.append(input_name)
                            input_node = self._get_input_node_for_edge(input_name)
                            to_calibrate.append((input_node, input_name))

                for calib_node, output_edge in to_calibrate:
                    (reduce_node, reduce_node_output,) = self.add_reduce_to_node_output(
                        calib_node, output_edge, "ReduceMin"
                    )
                    added_nodes.append(reduce_node)
                    added_outputs.append(reduce_node_output)

                    (reduce_node, reduce_node_output,) = self.add_reduce_to_node_output(
                        calib_node, output_edge, "ReduceMax"
                    )
                    added_nodes.append(reduce_node)
                    added_outputs.append(reduce_node_output)

        # use optimized model if available
        base_model_path = self._optimized_model_path or self._onnx_file
        augmented_model = onnx.load(base_model_path)
        augmented_model.graph.node.extend(added_nodes)
        augmented_model.graph.output.extend(added_outputs)
        return augmented_model

    def _iter_calib_ops_output(
        self,
        outputs: List[np.ndarray],
    ) -> Generator[Tuple[str, float, float], None, None]:
        """
        :param outputs: the outputs of a run of the augmented model
        :return: A generator that for every augmented operation yields
            the operation name, the value of the REDUCE_MIN operator,
            and the value of the REDUCE_MAX operator associated with
            the operation.
        """
        num_orig_outputs = len(self._model.graph.output)
        output_names = [
            output_obj.name for output_obj in self._model_augmented.graph.output
        ]

        calib_output_names = output_names[num_orig_outputs:]
        calib_outputs = outputs[num_orig_outputs:]

        # Iterate through outputs in pairs of min, max
        assert len(calib_output_names) % 2 == 0
        for idx in range(0, len(calib_output_names), 2):
            min_op_name = calib_output_names[idx]
            max_op_name = calib_output_names[idx + 1]
            base_op_name = min_op_name.split("_Reduce")[0]

            # Check that the pairs match and min and max ops are in the right order
            assert "ReduceMin" in min_op_name
            assert "ReduceMax" in max_op_name
            if base_op_name != max_op_name.split("_Reduce")[0]:
                raise RuntimeError(
                    "Unexpected reduce output pair: {}, {}".format(
                        min_op_name, max_op_name
                    )
                )

            yield base_op_name, calib_outputs[idx], calib_outputs[idx + 1]

    def process_batch(self, input_batch: Dict[str, np.ndarray]) -> None:
        """
        Updates the model's calibration thresholds based on a run of the input batch

        :param input_batch: Dictionary of pre-processed model input batch to use, with
            input names mapped to a numpy array of the batch
        """
        batch_size = list(input_batch.values())[0].shape[0]
        if batch_size not in self._sessions:
            self._sessions[batch_size] = ORTModelRunner(
                self._augmented_model_path, batch_size=batch_size
            )
        outputs, _ = self._sessions[batch_size].batch_forward(input_batch)
        # extract just output values from ordered dict
        outputs = list(outputs.values())

        for op_name, min_val, max_val in self._iter_calib_ops_output(outputs):
            if op_name not in self._quantization_thresholds:
                self._quantization_thresholds[op_name] = (min_val, max_val)
            else:
                op_prev_min, op_prev_max = self._quantization_thresholds[op_name]
                self._quantization_thresholds[op_name] = (
                    min(op_prev_min, min_val),
                    max(op_prev_max, max_val),
                )

    def get_quantization_params_dict(self) -> Dict[str, List[Union[int, float]]]:
        """
        :return: A dictionary of quantization parameters based on the original
            model and calibrated quantization thresholds from runs of the
            process_batch function.  The format of the dictionary will be:
            {"param_name": [zero_point, scale]}
        """
        quantization_params = {}
        for idx, node in enumerate(self._model.graph.node):
            node_output_name = node.output[0]
            if node_output_name in self._quantization_thresholds:
                range_min, range_max = self._quantization_thresholds[node_output_name]
                next_nodes = get_node_output_nodes(self._model, node)
                # only pass next_node for optimization if there is 1
                next_node = next_nodes[0] if len(next_nodes) == 1 else None
                node_params = CalibrationSession._calculate_scale_zeropoint(
                    range_min, range_max, next_node
                )
                quantization_params[node_output_name] = node_params
        # Add model inputs to quantization_params
        for input_name in self.get_model_input_names():
            if (
                input_name in self._quantization_thresholds
                and input_name not in quantization_params
            ):
                range_min, range_max = self._quantization_thresholds[input_name]
                inp_params = CalibrationSession._calculate_scale_zeropoint(
                    range_min, range_max, None
                )
                quantization_params[input_name] = inp_params

        return quantization_params

    @staticmethod
    def _calculate_scale_zeropoint(
        range_min: float,
        range_max: float,
        next_node: Union[None, onnx.NodeProto],
    ) -> List[Union[int, float]]:
        # adjust range_min and range_max such that 0 is included in the range.
        # to make sure zero can be uniquely represented.
        range_min = min(range_min, 0)
        range_max = max(range_max, 0)

        # We update the output range min and max when next node is clip or relu
        # With this technique we can remove these 2 ops and
        # reduce the output range which in turn helps to improve accuracy
        if next_node is not None:
            if next_node.op_type == "Clip":
                clip_min = next_node.attribute[0].f
                clip_max = next_node.attribute[1].f
                if range_min < clip_min:
                    range_min = clip_min
                if range_max > clip_max:
                    range_max = clip_max
            if next_node.op_type == "Relu":
                if range_min < 0:
                    range_min = 0

        scale = np.float32(
            (range_max - range_min) / 255 if range_min != range_max else 1
        )
        initial_zero_point = (0 - range_min) / scale
        zero_point = np.uint8(round(max(0, min(255, initial_zero_point))))

        return [zero_point, scale]

    def __del__(self):
        """
        Cleans up any unnecessary files.
        """
        if self._optimized_model_path is not None:
            os.remove(self._optimized_model_path)
