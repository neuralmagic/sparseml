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

import pytest
from onnx import AttributeProto, TensorProto, helper

from sparseml.exporters.transforms import OnnxTransform
from sparsezoo.utils import save_onnx


def _create_model():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])
    pads = helper.make_tensor_value_info("pads", TensorProto.FLOAT, [1, 4])
    value = helper.make_tensor_value_info("value", AttributeProto.FLOAT, [1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])
    node_def = helper.make_node(
        "Pad",
        ["X", "pads", "value"],
        ["Y"],
        mode="constant",
    )
    graph_def = helper.make_graph(
        [node_def],
        "test-model",
        [X, pads, value],
        [Y],
    )
    return helper.make_model(graph_def, producer_name="onnx-example")


class _TestOnnxTransform(OnnxTransform):
    def transform(self, model):
        return model


def test_onnx_transform_from_invalid_path():
    transform = _TestOnnxTransform()
    with pytest.raises(FileNotFoundError):
        transform("invalid_path/onnx_model.onnx")


def test_onnx_transform_from_model():
    model = _create_model()
    transform = _TestOnnxTransform()
    assert transform(model)


def test_onnx_transform_from_path(tmp_path):
    model = _create_model()
    transform = _TestOnnxTransform()
    path = os.path.join(str(tmp_path), "model.onnx")
    save_onnx(model, path)
    assert transform(path)
    os.remove(path)
