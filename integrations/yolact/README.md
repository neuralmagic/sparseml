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

# SparseML YOLACT Integration

This directory combines the [SparseML] recipe-driven approach with the 
[dbolya/yolact](https://github.com/dbolya/yolact) repository.
By integrating the training flows in the `yolact` repository with the SparseML 
code base,
we enable model sparsification techniques on the popular 
[YOLACT architecture](https://arxiv.org/abs/1804.02767)
creating smaller and faster deployable versions.
The techniques include, but are not limited to:

- Pruning
- Quantization
- Pruning + Quantization
- Sparse Transfer Learning

## Installation

To begin, run the following command in the root directory of this integration 
(`cd integrations/yolact`):

```bash
bash setup_integration.sh
```

### Exporting for Inference

After sparsifying a model, the 
[`export.py` script](https://github.com/neuralmagic//blob/master/export.py)
converts the model into deployment formats such as [ONNX](https://onnx.ai/).
The export process is modified such that the quantized and pruned models are 
corrected and folded properly.

For example, the following command can be run from within the neuralmagic's 
yolact repository folder to export a trained/sparsified model's checkpoint:
```bash
python export.py --checkpoint ./quantized-yolact/yolact_darknet53_0_10.pth \
    --recipe ./recipes/yolact.quant.yaml \
    --save-dir ./exported-test \
    --name quantized-yolact --batch-size 1 \
    --image-shape 3 550 550 \
    --config yolact_darknet53_config
```

To prevent conversion of a QAT(Quantization Aware Training) Graph to a
Quantized Graph, pass in the `--no-qat` flag:

```bash
python export.py --checkpoint ./quantized-yolact/yolact_darknet53_0_10.pth \
    --recipe ./recipes/yolact.quant.yaml \
    --save-dir ./exported-test \
    --name qat-yolact --batch-size 1 \
    --image-shape 3 550 550 \
    --config yolact_darknet53_config \
    --no-qat
```

The [DeepSparse](https://github.com/neuralmagic/deepsparse) Engine accepts ONNX 
formats and is engineered to significantly speed up inference on CPUs for 
the sparsified models from this integration.