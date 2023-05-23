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
import tempfile

import numpy
import onnx
import pytest

from sparseml.onnx.optim.quantization.quantize_model_post_training import (
    quantize_model_post_training,
)
from sparseml.onnx.utils import (
    DataLoader,
    ORTModelRunner,
    quantize_resnet_identity_add_inputs,
)
from sparseml.pytorch.datasets import ImagenetteDataset, ImagenetteSize
from sparsezoo import Model
from sparsezoo.utils import save_onnx


def _test_model_is_quantized(
    base_model_path, quant_model_path, excluded_convs=0, excluded_matmuls=0
):
    base_model = onnx.load(base_model_path)
    quant_model = onnx.load(quant_model_path)

    n_convs = len([n for n in base_model.graph.node if n.op_type == "Conv"])
    n_matmuls = len([n for n in base_model.graph.node if n.op_type == "MatMul"])

    n_qconvs = len([n for n in quant_model.graph.node if n.op_type == "QLinearConv"])
    n_qmatmuls = len(
        [n for n in quant_model.graph.node if n.op_type == "QLinearMatMul"]
    )

    assert n_convs - excluded_convs == n_qconvs
    assert n_matmuls - excluded_matmuls == n_qmatmuls


def _test_quant_model_output(
    base_model_path, quant_model_path, data_loader, test_output_idxs, batch_size
):
    base_sess = ORTModelRunner(base_model_path, batch_size=batch_size)
    quant_sess = ORTModelRunner(quant_model_path, batch_size=batch_size)
    # Test 100 random samples, aiming for > 98 matches
    n_matches = 0
    for i in range(100):
        # Generate and run a random sample batch
        sample_batch, _ = next(data_loader)
        base_outputs, _ = base_sess.batch_forward(sample_batch)
        quant_outputs, _ = quant_sess.batch_forward(sample_batch)

        base_outputs = list(base_outputs.values())
        quant_outputs = list(quant_outputs.values())
        # Check that the predicted values of outputs are the same
        for idx in test_output_idxs:
            if numpy.argmax(base_outputs[idx]) == numpy.argmax(quant_outputs[idx]):
                n_matches += 1
    # check that at least 98% match, should be higher in practice
    assert n_matches >= 98 * len(test_output_idxs)


def _test_resnet_identity_quant(model_path, has_resnet_block, save_optimized):
    quant_model = onnx.load(model_path)
    if has_resnet_block:  # run ResNet optimization
        assert quantize_resnet_identity_add_inputs(quant_model)
    # check that running the optimization has no affect even if its already been run
    assert not quantize_resnet_identity_add_inputs(quant_model)
    if save_optimized:
        save_onnx(quant_model, model_path)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_QUANTIZATION_TESTS", False),
    reason="Skipping quantization tests",
)
def test_quantize_model_post_training_resnet50_imagenette():
    # Prepare model paths
    resnet50_imagenette_path = Model(
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenette/base-none"
    ).onnx_model.path
    quant_model_path = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name

    # Prepare sample validation dataset
    batch_size = 1
    val_dataset = ImagenetteDataset(train=False, dataset_size=ImagenetteSize.s320)
    input_dict = [{"input": img.numpy()} for (img, _) in val_dataset]
    data_loader = DataLoader(input_dict, None, batch_size)

    # Run calibration and quantization
    quantize_model_post_training(
        resnet50_imagenette_path,
        data_loader,
        quant_model_path,
        show_progress=False,
        run_extra_opt=False,
    )

    # Verify that ResNet identity optimization is successful and save output for testing
    _test_resnet_identity_quant(quant_model_path, True, True)

    # Verify Convs and MatMuls are quantized
    _test_model_is_quantized(resnet50_imagenette_path, quant_model_path)

    # Verify quant model accuracy
    test_data_loader = DataLoader(input_dict, None, 1)  # initialize a new generator
    _test_quant_model_output(
        resnet50_imagenette_path, quant_model_path, test_data_loader, [1], batch_size
    )

    # Clean up
    os.remove(quant_model_path)
