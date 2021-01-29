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
import onnx
import onnxruntime as rt
import pytest

from sparseml.onnx.utils.graph_optimizer import fold_conv_bns
from tests.sparseml.onnx.optim.quantization.helpers import (
    make_tmp_onnx_file,
    onnx_conv_net,
)


def _model_has_conv_bn(model: onnx.ModelProto):
    conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
    for conv_node in conv_nodes:
        conv_output = conv_node.output[0]
        child_nodes = [n for n in model.graph.node if conv_output in n.input]
        for child_node in child_nodes:
            if child_node.op_type == "BatchNormalization":
                return True
    return False


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_QUANTIZATION_TESTS", False),
    reason="Skipping quantization tests",
)
@pytest.mark.parametrize(
    "model_lambda, inputs_dtype",
    [(onnx_conv_net, np.float32)],
)
def test_fold_conv_bn(model_lambda, inputs_dtype):
    base_model = model_lambda()
    base_model_path = make_tmp_onnx_file(base_model)

    # Test that there is a conv -> batch norm op in the model
    assert _model_has_conv_bn(base_model)

    # Fold conv-bns
    model_folded = fold_conv_bns(base_model_path)
    assert model_folded is not None
    folded_model_path = make_tmp_onnx_file(model_folded)

    # Check that there are no conv -> batch norm ops in the model
    assert not _model_has_conv_bn(model_folded)

    # Check that the outputs of the original and optimized graphs are equal
    base_sess = rt.InferenceSession(base_model_path)
    base_input_names = [inp.name for inp in base_sess.get_inputs()]
    base_input_shapes = [inp.shape for inp in base_sess.get_inputs()]
    base_output_names = [out.name for out in base_sess.get_outputs()]

    folded_sess = rt.InferenceSession(folded_model_path)
    folded_input_names = [inp.name for inp in folded_sess.get_inputs()]
    folded_input_shapes = [inp.shape for inp in folded_sess.get_inputs()]
    folded_output_names = [out.name for out in folded_sess.get_outputs()]

    assert base_input_names == folded_input_names
    assert base_input_shapes == folded_input_shapes
    assert len(base_output_names) == len(folded_output_names)

    for i in range(5):  # Test 5 random inputs for consistency
        inputs = [
            np.random.randn(*shape).astype(inputs_dtype) for shape in base_input_shapes
        ]
        base_outputs = base_sess.run(
            base_output_names, dict(zip(base_input_names, inputs))
        )
        folded_outputs = folded_sess.run(
            folded_output_names, dict(zip(folded_input_names, inputs))
        )
        for base_out, folded_out in zip(base_outputs, folded_outputs):
            assert np.max(np.abs(base_out - folded_out)) <= 1e-4

    # Clean up tmp files
    os.remove(base_model_path)
    os.remove(folded_model_path)
