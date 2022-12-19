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
from collections import Counter

import numpy
import onnx
import onnxruntime as ort
import pytest
import torch

from sparseml.exporters.onnx_to_deepsparse import ONNXToDeepsparse
from sparseml.pytorch.models.registry import ModelRegistry
from sparseml.pytorch.sparsification.quantization import QuantizationModifier
from sparseml.pytorch.torch_to_onnx_exporter import TorchToONNX
from sparseml.pytorch.utils import ModuleExporter
from tests.sparseml.pytorch.helpers import ConvNet, LinearNet, MLPNet


@pytest.mark.parametrize(
    "model,sample_batch",
    [
        (MLPNet(), torch.randn(8)),
        (MLPNet(), torch.randn(10, 8)),
        (LinearNet(), torch.randn(8)),
        (LinearNet(), torch.randn(10, 8)),
        (ConvNet(), torch.randn(1, 3, 28, 28)),
    ],
)
def test_simple_models_against_module_exporter(tmp_path, model, sample_batch):
    old_exporter = ModuleExporter(model, tmp_path / "old_exporter")
    old_exporter.export_onnx(sample_batch, convert_qat=False)

    (tmp_path / "new_exporter").mkdir()
    new_exporter = TorchToONNX(sample_batch)
    new_exporter.export(model, tmp_path / "new_exporter" / "model.onnx")

    _assert_onnx_models_are_equal(
        str(tmp_path / "old_exporter" / "model.onnx"),
        str(tmp_path / "new_exporter" / "model.onnx"),
        sample_batch,
    )


@pytest.mark.skipif(
    not os.getenv("RUN_EXPORTER_REGRESSION_TESTS", False),
    reason="Slow regression tests",
)
@pytest.mark.parametrize(
    "quantize,convert_qat",
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
def test_resnet50_exporters_are_equivalent(tmp_path, quantize: bool, convert_qat: bool):
    old_dir = tmp_path / "old_exporter"
    old_dir.mkdir()
    new_dir = tmp_path / "new_exporter"
    new_dir.mkdir()

    sample_batch = torch.rand(10, 3, 224, 224)
    model = ModelRegistry.create(
        "resnet50",
        pretrained=True,
        pretrained_path=(
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none"
        ),
    )
    if quantize:
        QuantizationModifier().apply(model)

    # TODO how to integrate this??
    use_qlinearconv = hasattr(model, "export_with_qlinearconv") and (
        model.export_with_qlinearconv
    )

    new_exporter = TorchToONNX(sample_batch)
    new_exporter.export(model, new_dir / "model.onnx")
    if convert_qat:
        ONNXToDeepsparse(use_qlinear_conv=use_qlinearconv).export(
            new_dir / "model.onnx", new_dir / "model.onnx"
        )

    old_exporter = ModuleExporter(model, old_dir)
    old_exporter.export_onnx(sample_batch, convert_qat=convert_qat)

    _assert_onnx_models_are_equal(
        str(tmp_path / "old_exporter" / "model.onnx"),
        str(tmp_path / "new_exporter" / "model.onnx"),
        sample_batch,
        expect_op_types=["QuantizeLinear", "DequantizeLinear"] if quantize else None,
    )


@pytest.mark.skipif(
    not os.getenv("RUN_EXPORTER_REGRESSION_TESTS", False),
    reason="Slow regression tests and requires yolov5 dependency",
)
def test_yolov5_exporters_are_equivalent(tmp_path):
    import shutil

    import sparseml.yolov5.scripts
    import yolov5.models.common
    from sparsezoo import Model

    class _HotPatchedBottleneck(torch.nn.Module):
        def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
            super().__init__()
            c_ = int(c2 * e)  # hidden channels
            self.cv1 = yolov5.models.common.Conv(c1, c_, 1, 1)
            self.cv2 = yolov5.models.common.Conv(c_, c2, 3, 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

    yolov5.models.common.Bottleneck = _HotPatchedBottleneck

    model = Model(
        "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",
        download_path=str(tmp_path / "stub"),
    )

    # first generate the pre-qat onnx file
    sparseml.yolov5.scripts.export(
        weights=model.training.path + "/model.pt",
        no_convert_qat=True,
        include=("onnx",),
    )
    shutil.move(
        model.training.path + "/model.onnx", model.training.path + "/model.preqat.onnx"
    )

    # now generate the full qat onnx file
    sparseml.yolov5.scripts.export(
        weights=model.training.path + "/model.pt",
        no_convert_qat=False,
        include=("onnx",),
    )

    sample_batch = torch.randint(0, 255, (2, 3, 640, 640), dtype=torch.uint8)

    ONNXToDeepsparse(use_qlinear_conv=True).export(
        model.training.path + "/model.preqat.onnx",
        tmp_path / "new_qat.onnx",
    )

    _assert_onnx_models_are_equal(
        model.training.path + "/model.onnx",
        str(tmp_path / "new_qat.onnx"),
        sample_batch,
        expect_op_types=["QuantizeLinear", "DequantizeLinear"],
    )


@pytest.mark.skipif(
    not os.getenv("RUN_EXPORTER_REGRESSION_TESTS", False),
    reason="Slow regression tests and requires yolov5 dependency",
)
@pytest.mark.parametrize(
    "quantize,convert_qat",
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
def test_bert_exporters_are_equivalent(tmp_path, quantize: bool, convert_qat: bool):

    from sparseml.transformers.utils import SparseAutoModel
    from sparsezoo import Model

    old_dir = tmp_path / "old_exporter"
    old_dir.mkdir()
    new_dir = tmp_path / "new_exporter"
    new_dir.mkdir()

    zoo_model = Model(
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none"
    )
    sample_batch = zoo_model.sample_inputs.sample_batch(batch_as_list=True)
    sample_batch = [torch.tensor(v) for v in sample_batch]

    torch_model = SparseAutoModel.question_answering_from_pretrained(
        model_name_or_path=zoo_model.training.path, model_type="model"
    )
    if quantize:
        QuantizationModifier().apply(torch_model.train())

    use_qlinearconv = hasattr(torch_model, "export_with_qlinearconv") and (
        torch_model.export_with_qlinearconv
    )
    new_exporter = TorchToONNX(sample_batch)
    new_exporter.export(torch_model, new_dir / "model.onnx")
    if convert_qat:
        ONNXToDeepsparse(use_qlinear_conv=use_qlinearconv).export(
            new_dir / "model.onnx", new_dir / "model.onnx"
        )

    old_exporter = ModuleExporter(torch_model, old_dir)
    old_exporter.export_onnx(sample_batch, convert_qat=convert_qat)

    _assert_onnx_models_are_equal(
        str(tmp_path / old_dir / "model.onnx"),
        str(tmp_path / new_dir / "model.onnx"),
        zoo_model.sample_inputs.sample_batch(batch_as_list=False),
        expect_op_types=["QuantizeLinear", "DequantizeLinear"] if quantize else None,
    )


def _assert_onnx_models_are_equal(
    old_model_path: str,
    new_model_path: str,
    sample_batch: torch.Tensor,
    expect_op_types=None,
):
    onnx.checker.check_model(old_model_path)
    onnx.checker.check_model(new_model_path)

    old_model = onnx.load(old_model_path)
    new_model = onnx.load(new_model_path)

    old_op_types = Counter([n.op_type for n in old_model.graph.node])
    new_op_types = Counter([n.op_type for n in new_model.graph.node])
    print("OLD", old_op_types)
    print("NEW", new_op_types)
    assert old_op_types == new_op_types
    if expect_op_types is not None:
        for op_type in expect_op_types:
            assert op_type in old_op_types
    assert len(old_model.graph.node) == len(new_model.graph.node)
    assert len(old_model.graph.initializer) == len(new_model.graph.initializer)

    old_session = ort.InferenceSession(old_model_path)
    input_name = old_session.get_inputs()[0].name
    output_name = old_session.get_outputs()[0].name
    old_output = old_session.run(
        [output_name],
        sample_batch
        if isinstance(sample_batch, dict)
        else {input_name: sample_batch.numpy()},
    )

    new_session = ort.InferenceSession(new_model_path)
    new_output = new_session.run(
        [output_name],
        sample_batch
        if isinstance(sample_batch, dict)
        else {input_name: sample_batch.numpy()},
    )

    assert numpy.allclose(old_output, new_output)
