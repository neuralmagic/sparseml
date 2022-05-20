# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

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
Export a YOLOv5 PyTorch model to other formats.
"""

import argparse
import os
import time
import warnings
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from sparseml.pytorch.sparsification.quantization import skip_onnx_input_quantize
from sparseml.pytorch.utils import ModuleExporter
from yolov5.models.common import Conv, DetectMultiBackend
from yolov5.models.experimental import attempt_load
from yolov5.models.yolo import Detect, Model
from yolov5.utils.activations import SiLU
from yolov5.utils.downloads import attempt_download
from yolov5.utils.general import (
    LOGGER,
    ROOT,
    check_img_size,
    check_requirements,
    colorstr,
    file_size,
    intersect_dicts,
    print_args,
    url2file,
)
from yolov5.utils.sparse import SparseMLWrapper, check_download_sparsezoo_weights
from yolov5.utils.torch_utils import (
    is_parallel,
    select_device,
    torch_distributed_zero_first,
)


FILE = Path(__file__).resolve()


def export_onnx(
    model, im, file, opset, train, dynamic, simplify, prefix=colorstr("ONNX:")
):
    # YOLOv5 ONNX export
    try:
        check_requirements(("onnx",))
        import onnx

        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
        f = file.with_suffix(".onnx")

        # export through SparseML so quantized and pruned graphs can be corrected
        save_dir = f.parent.absolute()
        save_name = str(f).split(os.path.sep)[-1]

        # get the number of outputs so we know how to name and change dynamic axes
        # nested outputs can be returned if model is exported with dynamic
        def _count_outputs(outputs):
            count = 0
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                for out in outputs:
                    count += _count_outputs(out)
            else:
                count += 1
            return count

        outputs = model(im)
        num_outputs = _count_outputs(outputs)
        input_names = ["input"]
        output_names = [f"out_{i}" for i in range(num_outputs)]
        dynamic_axes = (
            {k: {0: "batch"} for k in (input_names + output_names)} if dynamic else None
        )
        exporter = ModuleExporter(model, save_dir)
        exporter.export_onnx(
            im,
            name=save_name,
            convert_qat=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        try:
            skip_onnx_input_quantize(f, f)
        except:  # noqa
            pass

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # LOGGER.info(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                check_requirements(("onnx-simplifier",))
                import onnxsim

                LOGGER.info(
                    f"{prefix} simplifying with onnx-simplifier "
                    f"{onnxsim.__version__}..."
                )
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={"images": list(im.shape)} if dynamic else None,
                )
                assert check, "assert check failed"
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f"{prefix} simplifier failure: {e}")
        LOGGER.info(f"{prefix} export success, saved as {f} ({file_size(f):.1f} MB)")
        return f
    except Exception as e:
        LOGGER.info(f"{prefix} export failure: {e}")


def create_checkpoint(epoch, model, optimizer, ema, sparseml_wrapper, **kwargs):
    pickle = not sparseml_wrapper.qat_active(
        epoch
    )  # qat does not support pickled exports
    ckpt_model = deepcopy(model.module if is_parallel(model) else model).float()
    yaml = ckpt_model.yaml
    if not pickle:
        ckpt_model = ckpt_model.state_dict()

    return {
        "epoch": epoch,
        "model": ckpt_model,
        "optimizer": optimizer.state_dict(),
        "yaml": yaml,
        "hyp": model.hyp,
        **ema.state_dict(pickle),
        **sparseml_wrapper.state_dict(),
        **kwargs,
    }


def load_checkpoint(
    type_,
    weights,
    device,
    cfg=None,
    hyp=None,
    nc=None,
    data=None,
    dnn=False,
    half=False,
    recipe=None,
    resume=None,
    rank=-1,
    one_shot=False,
):
    with torch_distributed_zero_first(rank):
        # download if not found locally or from sparsezoo if stub
        weights = attempt_download(weights) or check_download_sparsezoo_weights(weights)
    ckpt = torch.load(
        weights[0]
        if isinstance(weights, list) or isinstance(weights, tuple)
        else weights,
        map_location="cpu",
    )  # load checkpoint
    start_epoch = ckpt["epoch"] + 1 if "epoch" in ckpt else 0
    pickled = isinstance(ckpt["model"], nn.Module)
    train_type = type_ == "train"
    ensemble_type = type_ == "ensemble"
    val_type = type_ == "val"

    if pickled and ensemble_type:
        cfg = None
        if ensemble_type:
            model = attempt_load(
                weights, map_location=device
            )  # load ensemble using pickled
            state_dict = model.state_dict()
        elif val_type:
            model = DetectMultiBackend(
                weights, device=device, dnn=dnn, data=data, fp16=half
            )
            state_dict = model.model.state_dict()
    else:
        # load model from config and weights
        cfg = (
            cfg
            or (ckpt["yaml"] if "yaml" in ckpt else None)
            or (ckpt["model"].yaml if pickled else None)
        )
        model = Model(
            cfg,
            ch=3,
            nc=ckpt["nc"] if ("nc" in ckpt and not nc) else nc,
            anchors=hyp.get("anchors") if hyp else None,
        ).to(device)
        model_key = (
            "ema" if (not train_type and "ema" in ckpt and ckpt["ema"]) else "model"
        )
        state_dict = (
            ckpt[model_key].float().state_dict() if pickled else ckpt[model_key]
        )
        if val_type:
            model = DetectMultiBackend(
                model=model, device=device, dnn=dnn, data=data, fp16=half
            )

    # turn gradients for params back on in case they were removed
    for p in model.parameters():
        p.requires_grad = True

    # load sparseml recipe for applying pruning and quantization
    checkpoint_recipe = train_recipe = None
    if resume:
        train_recipe = ckpt.get("recipe")
    elif recipe or ckpt.get("recipe"):
        train_recipe, checkpoint_recipe = recipe, ckpt.get("recipe")

    sparseml_wrapper = SparseMLWrapper(
        model.model if val_type else model,
        checkpoint_recipe,
        train_recipe,
        one_shot=one_shot,
    )
    exclude_anchors = train_type and (cfg or hyp.get("anchors")) and not resume
    loaded = False

    sparseml_wrapper.apply_checkpoint_structure()
    if train_type:
        # intialize the recipe for training and restore the weights before
        # if no quantized weights
        quantized_state_dict = any(
            [name.endswith(".zero_point") for name in state_dict.keys()]
        )
        if not quantized_state_dict:
            state_dict = load_state_dict(
                model, state_dict, train=True, exclude_anchors=exclude_anchors
            )
            loaded = True
        if not one_shot:
            sparseml_wrapper.initialize(start_epoch)

    if not loaded:
        state_dict = load_state_dict(
            model, state_dict, train=train_type, exclude_anchors=exclude_anchors
        )

    model.float()
    report = "Transferred %g/%g items from %s" % (
        len(state_dict),
        len(model.state_dict()),
        weights,
    )

    if val_type:
        model.model.eval()

    return model, {
        "ckpt": ckpt,
        "state_dict": state_dict,
        "sparseml_wrapper": sparseml_wrapper,
        "report": report,
    }


def load_state_dict(model, state_dict, train, exclude_anchors):
    # fix older state_dict names not porting to the new model setup
    state_dict = {
        key if not key.startswith("module.") else key[7:]: val
        for key, val in state_dict.items()
    }

    if train:
        # load any missing weights from the model
        state_dict = intersect_dicts(
            state_dict,
            model.state_dict(),
            exclude=["anchor"] if exclude_anchors else [],
        )

    model.load_state_dict(state_dict, strict=not train)  # load

    return state_dict


def export_formats():
    x = [
        ["PyTorch", "-", ".pt", True],
        ["TorchScript", "torchscript", ".torchscript", True],
        ["ONNX", "onnx", ".onnx", True],
        ["OpenVINO", "openvino", "_openvino_model", False],
        ["TensorRT", "engine", ".engine", True],
        ["CoreML", "coreml", ".mlmodel", False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True],
        ["TensorFlow GraphDef", "pb", ".pb", True],
        ["TensorFlow Lite", "tflite", ".tflite", False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False],
        ["TensorFlow.js", "tfjs", "_web_model", False],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "GPU"])


@torch.no_grad()
def run(
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=(640, 640),  # image (height, width)
    batch_size=1,  # batch size
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=False,  # FP16 half-precision export
    inplace=False,  # set YOLOv5 Detect() inplace=True
    train=False,  # model.train() mode
    dynamic=False,  # ONNX/TF: dynamic axes
    simplify=False,  # ONNX: simplify model
    opset=12,  # ONNX: opset version
    remove_grid=False,
):
    t = time.time()
    file = Path(
        url2file(weights) if str(weights).startswith(("http:/", "https:/")) else weights
    )  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    assert not (
        device.type == "cpu" and half
    ), "--half only compatible with GPU export, i.e. use --device 0"
    model, extras = load_checkpoint(
        type_="ensemble", weights=weights, device=device
    )  # load FP32 model
    nc, names = (
        extras["ckpt"].get("nc") or model.nc,
        model.names,
    )  # number of classes, class names

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    assert nc == len(names), f"Model class count {nc} != len(names) {len(names)}"

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(
        device
    )  # image size(1,3,320,192) BCHW iDetection

    # Update model
    if half:
        im, model = im.half(), model.half()  # to FP16
    # training mode = no Detect() layer grid construction
    model.train() if train else model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            if hasattr(m, "forward_export"):
                m.forward = m.forward_export  # assign custom forward (optional)
    model.model[-1].export = not remove_grid  # set Detect() layer grid export

    for _ in range(2):
        y = model(im)  # dry runs
    shape = tuple(y[0].shape)  # model output shape
    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} "
        f"({file_size(file):.1f} MB)"
    )

    warnings.filterwarnings(
        action="ignore", category=torch.jit.TracerWarning
    )  # suppress TracerWarning
    f = export_onnx(model, im, file, opset, train, dynamic, simplify)

    # Finish
    if f:
        LOGGER.info(
            f"\nExport complete ({time.time() - t:.2f}s)"
            f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
            f"\nVisualize:       https://netron.app"
        )
    return f  # return exported onnx path file


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolov5s.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="image (h, w)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--half", action="store_true", help="FP16 half-precision export"
    )
    parser.add_argument(
        "--inplace", action="store_true", help="set YOLOv5 Detect() inplace=True"
    )
    parser.add_argument("--train", action="store_true", help="model.train() mode")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF: dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=12, help="ONNX: opset version")
    parser.add_argument(
        "--remove-grid",
        action="store_true",
        help="remove export of Detect() layer grid",
    )
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt=None):
    if opt is None:
        opt = parse_opt()
    for opt.weights in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
