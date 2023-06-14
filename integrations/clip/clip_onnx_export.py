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
Examples on how to use sparsemls's onnx export functionality to export CLIP visual and text models
using the OpenCLIP API.

Note: This requires torch nightly and openclip to be installed:
https://github.com/mlfoundations/open_clip

"""
import argparse
from pathlib import Path
from typing import Any, Union

import torch
from torchvision.transforms.transforms import Compose

import open_clip
from clip_models import TextModel, VisualModel
from sparseml.pytorch.utils import export_onnx


def _export_onnx(
    module: torch.nn.Module,
    sample_batch: Any,
    file_path: Path,
    opset: int = 14,
    **export_kwargs,
):
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
    transformations: Compose,
    **export_kwargs,
):
    module_name = "clip_visual.onnx"
    visual_model = VisualModel(
        visual_model=model.visual, transformations=transformations
    )

    image_shape = visual_model.visual_model.image_size[0]
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
    **export_kwargs,
):
    module_name = "clip_text.onnx"
    text_model = TextModel(
        token_embedding=model.token_embedding,
        tokenizer=tokenizer,
        positional_embedding=model.positional_embedding,
        transformer=model.transformer,
        ln_final=model.ln_final,
        text_projection=model.text_projection,
        attn_mask=model.attn_mask,
    )

    text_model = text_model.to(device)
    text_model.eval()

    sample_batch = tokenizer(["a diagram", "a dog", "a cat"])
    _export_onnx(
        module=text_model,
        sample_batch=sample_batch,
        file_path=export_path / module_name,
        **export_kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch CLIP models and export to onnx using sparseml"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-16-plus-240",
        help="Name of CLIP model. See open_clip docs for a list of available models",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion400m_e32",
        help="Name of the pretraining to use. See open_clip docs for a list of available options.",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default="clip_onnx",
        help="Path of the directory to which the onnx outputs will be exported to",
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

    model, _, transform = open_clip.create_model_and_transforms(
        model_name=args.model, pretrained=args.pretrained
    )

    tokenizer = open_clip.get_tokenizer(args.model)

    _export_visual(model, device, clip_onnx_path, transform, **export_kwargs)
    _export_text(model, device, clip_onnx_path, tokenizer, **export_kwargs)


if __name__ == "__main__":
    main()
