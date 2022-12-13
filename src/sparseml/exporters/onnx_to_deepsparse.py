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

from typing import Any

import onnx

from sparseml.exporters import transforms as sparseml_transforms
from sparseml.exporters.base_exporter import BaseExporter


class ONNXToDeepsparse(BaseExporter):
    """
    Optimizes an `onnx.ModelProto` for the deepsparse engine by applying a
    series of transformations to a onnx graph with quantize operations.

    Usage:
    ```python
    model: onnx.ModelProto = ... # could be from TorchToOnnx() or somewhere else
    exporter = ONNXToDeepsparse()
    exporter.export(model, "model.onnx")
    ```

    You can also just optimize the model directly without saving to disk:
    ```python
    model: onnx.ModelProto = ...
    exporter = ONNXToDeepsparse()
    optimized_model = exporter.apply(model)
    ```

    :param use_qlinearconv: Set True to use legacy QLinearConv format instead
        of ConvInteger. QLinearConv requires output activations be quantized
        in the quantization recipe. (This was the default behavior prior to
        sparseml 0.12). Default is False
    :param skip_input_quantize: if True, the export flow will attempt to delete
        the first Quantize Linear Nodes(s) immediately after model input and set
        the model input type to UINT8. Default is False
    """

    def __init__(
        self,
        use_qlinear_conv: bool = False,
        skip_input_quantize: bool = False,
    ):
        cleanups = [
            sparseml_transforms.ConstantsToInitializers(),
            sparseml_transforms.FoldIdentityInitializers(),
            sparseml_transforms.InitializersToUint8(),
            sparseml_transforms.FoldConvDivBn(),
            sparseml_transforms.FoldReLUQuants(),
            sparseml_transforms.QuantizeQATEmbedding(),
            sparseml_transforms.PropagateEmbeddingQuantization(),
        ]

        if skip_input_quantize:
            cleanups.append(sparseml_transforms.SkipInputQuantize())

        qat_transforms = [
            sparseml_transforms.ConvertQuantizableMatmul(),
            sparseml_transforms.MatMulToMatMulIntegerAddCastMul(),
            sparseml_transforms.GemmToQLinearMatMul(),
            sparseml_transforms.GemmToMatMulIntegerAddCastMul(),
            sparseml_transforms.ConvToQLinearConv()
            if use_qlinear_conv
            else sparseml_transforms.ConvertQuantizableConvInteger(),
        ]

        reductions = [
            sparseml_transforms.RemoveDuplicateQConvWeights(),
            sparseml_transforms.QuantizeResiduals(),
            sparseml_transforms.RemoveDuplicateQuantizeOps(),
        ]

        super().__init__(cleanups + qat_transforms + reductions)

    def pre_validate(self, model: Any) -> onnx.ModelProto:
        if not isinstance(model, onnx.ModelProto):
            raise TypeError(f"Expected onnx.ModelProto, found {type(model)}")
        return model

    def post_validate(self, model: Any) -> onnx.ModelProto:
        # sanity check
        assert isinstance(model, onnx.ModelProto)
        return model

    def export(self, pre_transforms_model: onnx.ModelProto, file_path: str):
        post_transforms_model: onnx.ModelProto = self.apply(pre_transforms_model)
        onnx.save(post_transforms_model, file_path)
