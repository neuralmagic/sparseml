from typing import Union
import torch
from pathlib import Path
import numpy
from ultralytics.yolo.utils import LOGGER
import os
__all__ = ["export_sample_inputs_outputs"]

def _graph_has_uint8_inputs(onnx_path: Union[str, Path]) -> bool:
    """
    Load onnx model and check if it's input is type 2 (unit8)
    """
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    return onnx_model.graph.input[0].type.tensor_type.elem_type == 2

def preprocess(batch, device, half = False):
    batch["img"] = batch["img"].to(device, non_blocking=True)
    batch["img"] = (batch["img"].half() if half else batch["img"].float()) / 255
    for k in ["batch_idx", "cls", "bboxes"]:
        batch[k] = batch[k].to(device)

    nb = len(batch["img"])

    return batch

def export_sample_inputs_outputs(
    data_loader,
    model: torch.nn.Module,
    save_dir: Path,
    device: str,
    number_export_samples: int,
    image_size: int,
    onnx_path: Union[str, Path, None] = None,
):
    """
    Export sample model input and output for testing with the DeepSparse Engine
    :param dataset: path to dataset to take samples from
    :param model: model to be exported. Used to generate outputs
    :param save_dir: directory to save samples to
    :param number_export_samples: number of samples to export
    :param image_size: image size
    :param onnx_path: Path to saved onnx model. Used to check if it uses uints8 inputs
    """

    LOGGER.info(
        f"Exporting {number_export_samples} sample model inputs and outputs for "
        "testing with the DeepSparse Engine"
    )

    exported_samples = 0

    # Sample export directories
    sample_in_dir = os.path.join(save_dir, "sample_inputs")
    sample_out_dir = os.path.join(save_dir, "sample_outputs")
    os.makedirs(sample_in_dir, exist_ok=True)
    os.makedirs(sample_out_dir, exist_ok=True)

    save_inputs_as_uint8 = _graph_has_uint8_inputs(onnx_path) if onnx_path else False
    model = model.to(device)
    model.eval()

    for batch in data_loader:
        # uint8 to float32, 0-255 to 0.0-1.0
        preprocessed_batch = preprocess(batch=batch, device=device)
        import matplotlib.pyplot as plt
        import numpy as np
        a = (preprocessed_batch["img"][0].detach().cpu().numpy() * 255).astype(np.uint8)
        plt.imshow(a.transpose(1,2,0))
        plt.show()
        model_out = model(preprocessed_batch["img"])

        # Move to cpu for exporting

        out1, _ = model_out
        sample_in = preprocessed_batch["img"].detach().to("cpu")
        sample_out = out1.detach().to("cpu")

        file_idx = f"{exported_samples}".zfill(4)

        # Save inputs as numpy array
        sample_input_filename = os.path.join(sample_in_dir,f"inp-{file_idx}.npz")
        if save_inputs_as_uint8:
            sample_in = (255 * sample_in).to(dtype=torch.uint8)
        numpy.savez(sample_input_filename, sample_in)

        # Save outputs as numpy array
        sample_output_filename = os.path.join(sample_out_dir, f"out-{file_idx}.npz")
        numpy.savez(sample_output_filename, sample_out)
        exported_samples += 1

        if exported_samples >= number_export_samples:
            break

    if exported_samples < number_export_samples:
        LOGGER.info(
            f"Could not export {number_export_samples} samples. Exhausted dataloader "
            f"and exported {exported_samples} samples",
            level="warning",
        )

    LOGGER.info(f"Complete export of {number_export_samples} to {save_dir}")