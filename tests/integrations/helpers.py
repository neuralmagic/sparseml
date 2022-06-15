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

import glob
import os
from collections import defaultdict
from typing import Callable, Optional

import numpy
import onnx
from onnxruntime import InferenceSession

from sparseml.onnx.utils import get_tensor_shape
from sparsezoo import Zoo


__all__ = [
    "get_configs_with_cadence",
    "model_op_counts_test",
    "model_inputs_outputs_test",
    "TEST_OPS",
]


def get_configs_with_cadence(cadence: str, dir_path: str = "."):
    """
    Find all config files in the given directory with a matching cadence.

    :param cadence: string signifying how often to run this test. Possible values are:
        commit, daily, weekly
    :param dir_path: path to the directory in which to search for the config files
    :return List of file paths to matching configs
    """
    all_files_found = glob.glob(os.path.join(dir_path, "configs", "test*.yaml"))
    matching_files = []
    for file in all_files_found:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("cadence:"):
                    if line.split(":")[1].strip().strip('"').lower() == cadence:
                        matching_files.append(file)
                        break
    return matching_files


"""
Network graph operations to include when comparing operation counts between models
"""
TEST_OPS = {
    "MatMul",
    "Gemm",
    "Conv",
    "MatMulInteger",
    "ConvInteger",
    "QLinearMatMul",
    "QLinearConv",
}


def model_op_counts_test(
    model_path_a: str,
    model_path_b: str,
):
    """
    Test that the number of operations of each type, are the same between two onnx
    models.

    :param model_path_a: path to one onnx model
    :param model_path_b: path to other onnx model
    """

    model_a = _load_onnx_model(model_path_a)
    model_b = _load_onnx_model(model_path_b)

    def _get_model_op_counts(model):
        op_counts = defaultdict(int)
        for node in model.graph.node:
            if node.op_type in TEST_OPS:
                op_counts[node.op_type] += 1
        return op_counts

    op_counts_a = _get_model_op_counts(model_a)
    op_counts_b = _get_model_op_counts(model_b)

    assert len(op_counts_a) > 0
    assert len(op_counts_a) == len(op_counts_b)

    for op, count_a in op_counts_a.items():
        assert op in op_counts_b
        assert count_a == op_counts_b[op]


def model_inputs_outputs_test(
    model_path_a: str,
    model_path_b: str,
    input_getter: Optional[Callable] = None,
    **input_getter_kwargs,
):
    """
    Test that the output generated by two onnx models is similar to within some error
    when given the same input

    :param model_path_a: path to one onnx model
    :param model_path_b: path to other onnx model
    :input_getter: optional function to replace generic input generation routine. To be
        used for models/integrations which don't take numpy arrays as input
    """
    # compare export and target graphs and build fake data
    model_a = _load_onnx_model(model_path_a)
    model_b = _load_onnx_model(model_path_b)
    assert len(model_a.graph.input) == len(model_b.graph.input)
    assert len(model_a.graph.output) == len(model_b.graph.output)

    sample_input = {}
    output_names = []

    if input_getter:
        sample_input = input_getter(**input_getter_kwargs)

    else:
        for input_a, input_b in zip(model_a.graph.input, model_b.graph.input):
            assert input_a.name == input_b.name
            input_a_shape = get_tensor_shape(input_a)
            assert input_a_shape == get_tensor_shape(input_b)
            sample_input[input_a.name] = numpy.random.randn(*input_a_shape).astype(
                numpy.float32
            )

    for output_a, output_b in zip(model_a.graph.output, model_b.graph.output):
        assert output_a.name == output_b.name
        assert get_tensor_shape(output_a) == get_tensor_shape(output_b)
        output_names.append(output_a.name)

    # run sample forward and test absolute max diff
    ort_sess_a = InferenceSession(model_path_a)
    ort_sess_b = InferenceSession(model_path_b)
    forward_output_a = ort_sess_a.run(output_names, sample_input)
    forward_output_b = ort_sess_b.run(output_names, sample_input)
    for out_a, out_b in zip(forward_output_a, forward_output_b):
        assert numpy.max(numpy.abs(out_a - out_b)) <= 1e-4


def _load_onnx_model(path: str):
    if path.startswith("zoo:"):
        model = Zoo.load_model_from_stub(path)
        model.download()
        path_onnx = model.onnx_file.downloaded_path()
    else:
        path_onnx = path

    return onnx.load(path_onnx)
