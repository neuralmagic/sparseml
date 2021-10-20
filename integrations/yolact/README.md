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

This directory combines the SparseML recipe-driven approach with the 
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