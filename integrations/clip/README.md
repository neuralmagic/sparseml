<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# CLIP Export Examples

The examples in `clip_onnx_export.py` provide the steps needed to export a CLIP model using sparseml's onnx exporting functionality. The models and pretrained weights are pulled in from [OpenClip](https://github.com/mlfoundations/open_clip/tree/main) and the command line tools provided allow exporting of a given model's Text and Visual branches. See the OpenClip repository for a full list of available models. For the CoCa/Caption models available in OpenClip, an additional text-decoder is also exported.

## Installation

The examples provided require `open_clip_torch==2.20.0` to be installed along with **torch nighly**. To work within the `sparseml` environment, be sure to set the environment variable `MAX_TORCH` to your installed version when 
installing torch nightly.

Steps:
- Install `sparseml[clip]`. This will ensure open_clip_torch is installed
- Uninstall torch by running:  
```
   pip uninstall torch
```
- Install torch nightly:  
```   
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/
```
- Set your environment variable to the correct torch version: Example: `export MAX_TORCH="2.1.0.dev20230613+cpu"`