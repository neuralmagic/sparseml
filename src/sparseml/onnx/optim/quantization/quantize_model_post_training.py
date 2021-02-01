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
Provides a wrapper function for calibrating and quantizing an Onnx model
"""

from typing import Iterable, List, Union

import onnx
from tqdm.auto import tqdm

from sparseml.onnx.optim.quantization.calibration import CalibrationSession
from sparseml.onnx.optim.quantization.quantize import QuantizationMode, quantize
from sparseml.onnx.utils import DataLoader, quantize_resnet_identity_add_inputs


__all__ = ["quantize_model_post_training"]


def quantize_model_post_training(
    onnx_file: str,
    data_loader: DataLoader,
    output_model_path: str = None,
    calibrate_op_types: Iterable[str] = ("Conv", "MatMul", "Gemm"),
    exclude_nodes: List[str] = None,
    include_nodes: List[str] = None,
    augmented_model_path: str = None,
    static: bool = True,
    symmetric_weight: bool = False,
    force_fusions: bool = False,
    show_progress: bool = True,
    run_extra_opt: bool = True,
) -> Union[None, onnx.ModelProto]:
    """
    Wrapper function for calibrating and quantizing an Onnx model

    :param onnx_file: File path to saved Onnx model to calibrate and quantize
    :param data_loader: Iterable of lists of model inputs or filepath to directory
        of numpy arrays. If the model has multiple inputs and an .npz file is
        provided, the function will try to extract each input from the .npz file
        by name.  If the names do not match, the function will try to extract the
        inputs in order.  Will raise an exception of the number of inputs does not
        match the number of arrays in the .npz file.
    :param output_model_path: Filepath to where the quantized model should be saved to.
        If not provided, then the quantized Onnx model object will be returned instead.
    :param calibrate_op_types: List of Onnx ops names to calibrate and quantize within
        the model. Currently Onnx only supports quantizing 'Conv' and 'MatMul' ops.
    :param exclude_nodes: List of operator names that should not be quantized
    :param include_nodes: List of operator names force to be quantized
    :param augmented_model_path: file path to save augmented model to for verification
    :param static: True to use static quantization. Default is static.
    :param symmetric_weight: True to use symmetric weight quantization.
        Default is False
    :param force_fusions: True to force fusions in quantization. Default is False
    :param show_progress: If true, will display a tqdm progress bar during calibration.
        Default is True
    :param run_extra_opt: If true, will run additional optimizations on the quantized
        model. Currently the only optimization is quantizing identity relu outputs in
        ResNet blocks
    :return: None or quantized onnx model object if output_model_path is not provided
    """
    calibrator = CalibrationSession(
        onnx_file,
        calibrate_op_types,
        exclude_nodes,
        include_nodes,
        augmented_model_path,
        static,
    )

    # data_loader must have a finite number of examples
    assert not data_loader.infinite

    data_iterator = tqdm(data_loader) if show_progress else data_loader

    for input_batch, _ in data_iterator:
        calibrator.process_batch(input_batch)

    quantization_params_dict = calibrator.get_quantization_params_dict()
    calibrated_quantized_model = quantize(
        calibrator.model,
        quantization_mode=QuantizationMode.QLinearOps,
        force_fusions=force_fusions,
        quantization_params=quantization_params_dict,
        nodes_to_exclude=exclude_nodes if exclude_nodes else None,
        symmetric_weight=symmetric_weight,
        static=static,
    )

    if run_extra_opt:
        quantize_resnet_identity_add_inputs(calibrated_quantized_model)

    if output_model_path is None:
        return calibrated_quantized_model
    else:
        onnx.save(calibrated_quantized_model, output_model_path)
