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

import os

import numpy as np
import pytest

from sparseml.onnx.optim.quantization.calibration import CalibrationSession
from tests.sparseml.onnx.optim.quantization.helpers import (
    make_tmp_onnx_file,
    onnx_conv_net,
    onnx_linear_net,
)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_QUANTIZATION_TESTS", False),
    reason="Skipping quantization tests",
)
@pytest.mark.parametrize(
    "model_lambda, static",
    [
        (onnx_conv_net, True),
        (onnx_conv_net, False),
        (onnx_linear_net, True),
        (onnx_linear_net, False),
    ],
)
def test_augmented_graph(model_lambda, static):
    model = model_lambda()
    model_path = make_tmp_onnx_file(model)

    calibrate_op_types = ["Conv", "MatMul"]
    calibrator = CalibrationSession(
        model_path, calibrate_op_types=calibrate_op_types, static=static
    )

    # Test Augmented Graph
    calibrator.model_augmented
    aug_model_nodes = [n.name for n in calibrator.model_augmented.graph.node]

    for node in calibrator.model.graph.node:
        if node.op_type in calibrate_op_types:
            min_op_name = "{}_ReduceMin".format(node.name)
            max_op_name = "{}_ReduceMax".format(node.name)
            assert min_op_name in aug_model_nodes
            assert max_op_name in aug_model_nodes

    # Check that inputs are processed in static mode
    if static:
        for inp in calibrator.model.graph.input:
            min_op_name = "{}_ReduceMin".format(inp.name)
            max_op_name = "{}_ReduceMax".format(inp.name)
            assert min_op_name in aug_model_nodes
            assert max_op_name in aug_model_nodes

    # Clean up
    os.remove(model_path)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_QUANTIZATION_TESTS", False),
    reason="Skipping quantization tests",
)
@pytest.mark.parametrize(
    "model_lambda, inputs_shape, inputs_dtype, input_names, static",
    [
        (onnx_conv_net, [(1, 3, 5, 5)], np.float32, ["input"], True),
        (onnx_conv_net, [(1, 3, 5, 5)], np.float32, ["input"], False),
        (onnx_linear_net, [(20, 20)], np.float32, ["input"], True),
        (onnx_linear_net, [(20, 20)], np.float32, ["input"], False),
    ],
)
def test_full_calibration_session(
    model_lambda, inputs_shape, inputs_dtype, input_names, static
):
    model = model_lambda()
    model_path = make_tmp_onnx_file(model)

    calibrate_op_types = ["Conv", "MatMul"]
    calibrator = CalibrationSession(
        model_path, calibrate_op_types=calibrate_op_types, static=static
    )

    # Run calibration
    for i in range(5):
        inp_batch = [
            np.random.rand(*shape).astype(inputs_dtype) for shape in inputs_shape
        ]
        inp_dict = dict(zip(input_names, inp_batch))
        calibrator.process_batch(inp_dict)

    # Test quantization params dict
    quant_params = calibrator.get_quantization_params_dict()
    num_matches_found = 0

    # check that quant_params has all the expected nodes
    for node in calibrator.model.graph.node:
        if node.op_type in calibrate_op_types:
            assert node.output[0] in quant_params
            num_matches_found += 1
    if static:  # Check input nodes are included in static
        for inp in calibrator.model.graph.input:
            assert inp.name in quant_params
            num_matches_found += 1

    # check that quant_params has only the expected nodes
    assert num_matches_found == len(quant_params)

    # check that all values in quant_params are valid
    for zero_pt, scale in quant_params.values():
        assert isinstance(zero_pt, np.uint8)
        assert isinstance(scale, np.float32)

    os.remove(model_path)
