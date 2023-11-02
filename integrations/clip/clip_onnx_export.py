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
Examples on how to use sparsemls's onnx export functionality to export CLIP visual
and text models using the OpenCLIP API.

Note: This requires torch nightly and openclip to be installed:
https://github.com/mlfoundations/open_clip

"""
import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Union

import torch

import open_clip
from clip_models import TextModel
from sparseml.pytorch.utils import export_onnx


def _export_onnx(
    module: torch.nn.Module,
    sample_batch: Any,
    file_path: Union[Path, str],
    opset: int = 14,
    **export_kwargs,
):
    # _export_onnx by default uses opset = 14 as required by CLIP and will fail
    # for opset < 14 as certain operators are not supported.
    if opset < 14:
        raise ValueError("CLIP onnx export requires a minimum opset of 14")

    export_onnx(
        module=module,
        sample_batch=sample_batch,
        opset=opset,
        file_path=file_path,
        **export_kwargs,
    )


def _export_visual(
    model: torch.nn.Module,
    device: str,
    export_path: Union[str, Path],
    is_coca: bool,
    **export_kwargs,
):
    module_name = "clip_visual.onnx"
    visual_model = model.visual

    image_shape = visual_model.image_size[0]
    sample_input = torch.randn(1, 3, image_shape, image_shape, requires_grad=True)

    visual_model = visual_model.to(device)
    visual_model.eval()

    _export_onnx(
        module=visual_model,
        sample_batch=sample_input,
        file_path=export_path / module_name,
        **export_kwargs,
    )


def _export_text(
    model: torch.nn.Module,
    device: str,
    export_path: Union[str, Path],
    tokenizer,
    is_coca: bool,
    **export_kwargs,
):
    module_name = "clip_text.onnx"
    # If the model is a CLIP CoCa model, store the text model as is. For non-CoCa
    # models, OpenCLIP does not provide access to the text model, only the transformer
    # therefore in that case, create a new TextModel object to wrap the transformer
    # and all relevant properties needed for the forward pass.
    if is_coca:
        text_model = model.text
    else:
        text_model = TextModel(
            token_embedding=model.token_embedding,
            positional_embedding=model.positional_embedding,
            transformer=model.transformer,
            ln_final=model.ln_final,
            text_projection=model.text_projection,
            attn_mask=model.attn_mask,
        )

    text_model = text_model.to(device)
    text_model.eval()

    if is_coca:
        sample_batch = torch.ones(6, 15, dtype=torch.long)
    else:
        sample_batch = tokenizer(["a dog"]).to(torch.int32)

    _export_onnx(
        module=text_model,
        sample_batch=sample_batch,
        file_path=export_path / module_name,
        **export_kwargs,
    )


def _export_text_decoder(
    model: torch.nn.Module, device: str, export_path: Union[str, Path], **export_kwargs
):

    module_name = "clip_text_decoder.onnx"
    decoder = model.text_decoder.to(device)
    decoder.eval()

    sample_batch = OrderedDict()
    sample_batch["image_embs"] = torch.randn(1, 255, model.text.output_dim)
    sample_batch["text_embs"] = torch.randn(1, 15, model.text.output_dim)

    _export_onnx(
        module=decoder,
        sample_batch=sample_batch,
        file_path=export_path / module_name,
        **export_kwargs,
    )


def main():
    """
    Given a model name and pretrained weights (see OpenClip for available options),
    the text and visual branches for CLIP are exported to onnx using sparseml's
    exporting functionality. Commandline tools are provided to export a specific model/
    pretraining however, by default, the visual and text branches of the ViT-B-32 model
    will be exported and saved to a directory called `clip_onnx`. A custom path can
    also be provided using the `export-path` argument. Custom names for the input and
    output nodes of the graph can also be assigned, using the `input_name` and
    `output_name` arguments.

    Specifically for CoCa models, an additional text-decoder is also exported and saved
    in the same folder. Currently, only coca_ViT-B-32 and coca_ViT-L-14 are supported.

    Example:
        python clip_onnx_export.py --model convnext_base_w_320 \
            --pretrained laion_aesthetic_s13b_b82k --export-path convnext_onnx

        ======== Diagnostic Run torch.onnx.export version 2.1.0.dev20230613+cpu ========
        verbose: False, log level: 40
        ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

        ======== Diagnostic Run torch.onnx.export version 2.1.0.dev20230613+cpu ========
        verbose: False, log level: 40
        ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

    """
    parser = argparse.ArgumentParser(
        description="Fetch CLIP models and export to onnx using sparseml"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="coca_ViT-B-32",
        help="Name of CLIP model. See OpenClip docs for a list of available models",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="mscoco_finetuned_laion2b_s13b_b90k",
        help="Name of the pretraining to use. See OpenClip docs for a list of options.",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default="clip_onnx",
        help="Path of the directory to which the onnx outputs will be saved.",
    )
    parser.add_argument(
        "--input_name",
        type=str,
        default="inputs",
        help="names to assign to the input nodes",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="outputs",
        help="names to assign to the output nodes",
    )

    args = parser.parse_args()

    device = "cpu"
    clip_onnx_path = Path(args.export_path)

    input_names = [args.input_name]
    output_names = [args.output_name]
    export_kwargs = {
        "input_names": input_names,
        "output_names": output_names,
        "do_constant_folding": True,
    }

    model, _, _ = open_clip.create_model_and_transforms(
        model_name=args.model, pretrained=args.pretrained
    )

    tokenizer = open_clip.get_tokenizer(args.model)
    is_coca = "coca" in args.model

    _export_visual(model, device, clip_onnx_path, is_coca, **export_kwargs)
    _export_text(model, device, clip_onnx_path, tokenizer, is_coca, **export_kwargs)

    if is_coca:
        _export_text_decoder(model, device, clip_onnx_path, **export_kwargs)


if __name__ == "__main__":
    main()
