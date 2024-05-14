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
from copy import deepcopy
from pathlib import Path
from typing import Union

import onnx
from onnx import ModelProto

from sparseml.exporters import transforms as sparseml_transforms
from sparseml.exporters.base_exporter import BaseExporter
from sparsezoo import save_onnx


class ONNXToDeepsparse(BaseExporter):
    """
    Optimizes an `onnx.ModelProto` for the deepsparse engine by applying a
    series of transformations to a onnx graph with quantize operations.

    Usage:
    ```python
    # could be a model retrieved previously from TorchToOnnx() or somewhere else
    onnx_model: onnx.ModelProto = ...
    exporter = ONNXToDeepsparse()
    exporter.export(onnx_model, "model.onnx")
    ```

    You can also just optimize the model directly without saving to disk:
    ```python
    onnx_model: onnx.ModelProto = ...
    exporter = ONNXToDeepsparse()
    optimized_model = exporter.apply(onnx_model)
    ```

    :param use_qlinearconv: Set True to use legacy QLinearConv format instead
        of ConvInteger. QLinearConv requires output activations be quantized
        in the quantization recipe. (This was the default behavior prior to
        sparseml 0.12). Default is False
    :param skip_input_quantize: if True, the export flow will attempt to delete
        the first Quantize Linear Nodes(s) immediately after model input and set
        the model input type to UINT8. Default is False
    :param inplace: If true, does conversion of model in place. Default is true
    :param export_input_model: If true, saves the input onnx model alongside the
        optimized model.
    """

    def __init__(
        self,
        use_qlinear_conv: bool = False,
        use_qlinear_matmul: bool = False,
        skip_input_quantize: bool = False,
        inplace: bool = True,
        export_input_model: bool = False,
    ):
        self.inplace = inplace
        self.export_input_model = export_input_model

        transforms = [
            sparseml_transforms.ConstantsToInitializers(),
            sparseml_transforms.FoldIdentityInitializers(),
            sparseml_transforms.InitializersToUint8(),
            sparseml_transforms.FlattenQParams(),
            sparseml_transforms.FoldConvDivBn(),
            sparseml_transforms.DeleteRepeatedQdq(),
            sparseml_transforms.QuantizeQATEmbedding(),
            sparseml_transforms.PropagateEmbeddingQuantization(),
            sparseml_transforms.PropagateDequantThroughSplit(),
        ]
        if use_qlinear_matmul:
            transforms.append(
                sparseml_transforms.MatMulToQLinearMatMul(),
            )

        transforms.extend(
            [
                sparseml_transforms.MatMulAddToMatMulIntegerAddCastMul(),
                sparseml_transforms.MatMulToMatMulIntegerCastMul(),
                sparseml_transforms.FoldReLUQuants(),
                sparseml_transforms.ConvToQLinearConv()
                if use_qlinear_conv
                else sparseml_transforms.ConvToConvIntegerAddCastMul(),
                sparseml_transforms.GemmToQLinearMatMul(),
                sparseml_transforms.GemmToMatMulIntegerAddCastMul(),
                sparseml_transforms.QuantizeResiduals(),
                sparseml_transforms.RemoveDuplicateQConvWeights(),
                sparseml_transforms.RemoveDuplicateQuantizeOps(),
            ]
        )

        if skip_input_quantize:
            transforms.append(sparseml_transforms.SkipInputQuantize())

        super().__init__(transforms)

    def pre_validate(self, model: Union[onnx.ModelProto, str, Path]) -> onnx.ModelProto:
        if isinstance(model, (str, Path)):
            model = onnx.load(str(model))

        if not isinstance(model, onnx.ModelProto):
            raise TypeError(f"Expected onnx.ModelProto, found {type(model)}")
        return model if self.inplace else deepcopy(model)

    def post_validate(self, model: onnx.ModelProto) -> onnx.ModelProto:
        # sanity check
        if not isinstance(model, onnx.ModelProto):
            raise TypeError(f"Expected onnx.ModelProto, found {type(model)}")
        return model

    def export(
        self,
        pre_transforms_model: Union[ModelProto, str],
        file_path: str,
        do_split_external_data: bool = True,
    ):
        if not isinstance(pre_transforms_model, ModelProto):
            pre_transforms_model = onnx.load(pre_transforms_model)
        if self.export_input_model or os.getenv("SAVE_PREQAT_ONNX", False):
            save_onnx(
                pre_transforms_model,
                file_path.replace(".onnx", ".preqat.onnx"),
                do_split_external_data=do_split_external_data,
            )

        post_transforms_model: onnx.ModelProto = self.apply(pre_transforms_model)
        save_onnx(
            post_transforms_model,
            file_path,
            do_split_external_data=do_split_external_data,
        )
