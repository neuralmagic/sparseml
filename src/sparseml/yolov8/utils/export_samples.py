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
from typing import Any, Dict

import numpy
import torch
from torch import device as device_class

from ultralytics.yolo.utils import LOGGER


__all__ = ["export_sample_inputs_outputs"]

# define the priority order for the execution providers
# prefer CUDA Execution Provider over CPU Execution Provider
EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def preprocess(
    batch: Dict[str, Any], device: device_class, half: bool = False
) -> Dict[str, Any]:
    """
    Ported from
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/v8/detect/val.py
    """
    batch["img"] = batch["img"].to(device, non_blocking=True)
    batch["img"] = (batch["img"].half() if half else batch["img"].float()) / 255
    for k in ["batch_idx", "cls", "bboxes"]:
        batch[k] = batch[k].to(device)
    return batch


@torch.no_grad()
def export_sample_inputs_outputs(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    save_dir: str,
    device: device_class,
    number_export_samples: int,
    onnx_path: str,
):
    """
    Export sample model input and output for testing with the DeepSparse Engine

    :param data_loader: path to data loader to take samples from
    :param model: model to be exported. Used to generate torch outputs
    :param save_dir: directory to save samples to
    :param device: device to run the inference (output generation) on
    :param number_export_samples: number of samples to export
    :param onnx_path: path to onnx model. Used to generate ORT outputs
    """
    try:
        import onnxruntime
    except (ImportError, ModuleNotFoundError) as exception:
        raise ValueError(
            "onnxruntime is needed to export samples for validation, but the "
            "module was  not found, try `pip install sparseml[onnxruntime]`"
        ) from exception

    LOGGER.info(
        f"Exporting {number_export_samples} sample model inputs and outputs for "
        "testing with the DeepSparse Engine"
    )

    exported_samples = 0

    # Sample export directories
    sample_in_dir = os.path.join(save_dir, "sample-inputs")
    sample_out_dir_torch = os.path.join(save_dir, "sample_outputs_torch")
    sample_out_dir_ort = os.path.join(save_dir, "sample_outputs_onnxruntime")

    os.makedirs(sample_in_dir, exist_ok=True)
    os.makedirs(sample_out_dir_torch, exist_ok=True)
    os.makedirs(sample_out_dir_ort, exist_ok=True)

    save_inputs_as_uint8 = _graph_has_uint8_inputs(onnx_path) if onnx_path else False

    # Prepare model for inference
    model = model.to(device)
    model.eval()

    # Prepare onnxruntime engine for inference
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=EP_list)

    LOGGER.info(f"Exporting sample inputs to directory {sample_in_dir}")
    LOGGER.info(f"Exporting sample torch outputs to directory {sample_out_dir_torch}")
    LOGGER.info(
        f"Exporting sample onnxruntime outputs to directory {sample_out_dir_ort}"
    )

    for batch in data_loader:
        file_idx = f"{exported_samples}".zfill(4)
        preprocessed_batch = preprocess(batch=batch, device=device)
        image = preprocessed_batch["img"]

        # Save torch outputs as numpy array
        _export_torch_outputs(image, model, sample_out_dir_torch, file_idx)

        # Convert input data type if needed
        if save_inputs_as_uint8:
            image = (255 * image).to(dtype=torch.uint8)

        # Save inputs as numpy array
        _export_inputs(image, sample_in_dir, file_idx)
        # Save onnxruntime outputs as numpy array
        _export_ort_outputs(
            image.cpu().numpy(), ort_session, sample_out_dir_ort, file_idx
        )

        exported_samples += 1

        if exported_samples >= number_export_samples:
            break

    if exported_samples < number_export_samples:
        LOGGER.info(
            f"Could not export {number_export_samples} samples. Exhausted dataloader "
            f"and exported {exported_samples} samples",
            level="warning",
        )

    LOGGER.info(
        f"Completed the export of {number_export_samples} "
        f"input/output samples to {save_dir}"
    )


def _export_torch_outputs(
    image: torch.Tensor, model: torch.nn.Module, sample_out_dir: str, file_idx: str
):

    # Run model to get torch outputs
    model_out = model(image)
    preds = model_out
    sample_output_filename = os.path.join(sample_out_dir, f"out-{file_idx}.npz")
    seg_prediction = None

    # Move to cpu for exporting
    # Segmentation currently supports two outputs
    if isinstance(preds, tuple):
        preds_out = preds[0].detach().to("cpu")
        seg_prediction = preds[1].detach().to("cpu")
    else:
        preds_out = preds.detach().to("cpu")

    numpy.savez(sample_output_filename, preds_out, seg_prediction=seg_prediction)


def _export_ort_outputs(
    image: numpy.ndarray,
    session: "onnxruntime.InferenceSession",  # noqa: F821
    sample_out_dir: str,
    file_idx: str,
):

    # Run model to get onnxruntime outputs
    ort_inputs = {session.get_inputs()[0].name: image}
    ort_outs = session.run(None, ort_inputs)
    preds = ort_outs
    seg_prediction = None

    if len(preds) > 1:
        preds_out = preds[0]
        seg_prediction = preds[1]
    else:
        preds_out = preds[0]

    preds_out = numpy.squeeze(preds_out, axis=0)

    sample_output_filename = os.path.join(sample_out_dir, f"out-{file_idx}.npz")
    numpy.savez(sample_output_filename, preds_out, seg_prediction=seg_prediction)


def _export_inputs(image: torch.Tensor, sample_in_dir: str, file_idx: str):

    sample_in = image.detach().to("cpu").squeeze(0)

    sample_input_filename = os.path.join(sample_in_dir, f"inp-{file_idx}.npz")
    numpy.savez(sample_input_filename, sample_in)


def _graph_has_uint8_inputs(onnx_path: str) -> bool:
    """
    Load onnx model and check if it's input is type 2 (unit8)
    """
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    return onnx_model.graph.input[0].type.tensor_type.elem_type == 2
