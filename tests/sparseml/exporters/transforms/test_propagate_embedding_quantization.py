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

import onnx
import pytest
from onnx import ModelProto

from sparseml.exporters.transforms.propagate_embedding_quantization import (
    PropagateEmbeddingQuantization,
)


@pytest.fixture
def onnx_model():
    """See docstring of transform"""
    model_input = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, (1,)
    )
    model_output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (1,)
    )
    scale = onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, (1,), [1.0])
    starts = onnx.helper.make_tensor("starts", onnx.TensorProto.INT64, (1,), [0])
    ends = onnx.helper.make_tensor("ends", onnx.TensorProto.INT64, (1,), [1])
    pads = onnx.helper.make_tensor("pads", onnx.TensorProto.INT64, (1,), [1])
    embeddings = onnx.helper.make_tensor(
        "embeddings", onnx.TensorProto.UINT8, (1,), [0]
    )
    gather = onnx.helper.make_node(
        "Gather", ["embeddings", "input"], ["gather_output"], name="gather"
    )
    dequant = onnx.helper.make_node(
        "DequantizeLinear",
        ["gather_output", "scale"],
        ["dequant_output"],
        name="dequant",
    )

    slice1 = onnx.helper.make_node(
        "Slice", ["dequant_output", "starts", "ends"], ["slice1_output"], name="slice1"
    )
    pad1 = onnx.helper.make_node(
        "Pad", ["slice1_output", "pads"], ["pad1_output"], name="pad1"
    )
    slice2 = onnx.helper.make_node(
        "Slice", ["dequant_output", "starts", "ends"], ["slice2_output"], name="slice2"
    )
    pad2 = onnx.helper.make_node(
        "Pad", ["slice2_output", "pads"], ["pad2_output"], name="pad2"
    )
    concat = onnx.helper.make_node(
        "Concat",
        ["pad1_output", "pad2_output", "dequant_output"],
        ["concat_output"],
        name="concat",
        axis=0,
    )

    graph = onnx.helper.make_graph(
        nodes=[gather, dequant, slice1, pad1, slice2, pad2, concat],
        name="g",
        inputs=[model_input],
        outputs=[model_output],
        initializer=[scale, starts, ends, embeddings, pads],
    )

    model = onnx.helper.make_model(graph)
    onnx.checker.check_model(model)
    return model


def test_vanilla(onnx_model: ModelProto):
    onnx_model = PropagateEmbeddingQuantization().apply(onnx_model)
    onnx.checker.check_model(onnx_model)

    assert [n.name for n in onnx_model.graph.node] == [
        "gather",
        "slice1",
        "slice2",
        "pad1",
        "pad2",
        "concat",
        "dequant",
    ]
