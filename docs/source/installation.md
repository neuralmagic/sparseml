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

# Installation

This repository is tested on Python 3.6-3.9, and Linux/Debian systems.
It is recommended to install in a [virtual environment](https://docs.python.org/3/library/venv.html) to keep your system in order.
Currently supported ML Frameworks are the following: `torch>=1.1.0,<=1.8.0`, `tensorflow>=1.8.0,<=2.0.0`, `tensorflow.keras >= 2.2.0`.

Install with pip using:

```bash
pip install sparseml
```

## Supported Framework Versions

The currently supported framework versions are:

- PyTorch-supported versions: `>= 1.1.0, < 1.8.0`
- Keras-supported versions: `2.3.0-tf` (through the TensorFlow `2.2` package; as of Feb 1st, 2021, `keras2onnx` has
not been tested for TensorFlow >= `2.3`). 
- TensorFlow V1-supported versions: >= `1.8.0` (TensorFlow >= `2.X` is not currently supported)

## Optional Dependencies

Additionally, optional dependencies can be installed based on the framework you are using.

PyTorch:

```bash
pip install sparseml[torch]
```

Keras:

```bash
pip install sparseml[tf_keras]
```

TensorFlow V1:

```bash
pip install sparseml[tf_v1]
```

TensorFlow V1 with GPU operations enabled:

```bash
pip install sparseml[tf_v1_gpu]
```

Depending on your device and CUDA version, you may need to install additional dependencies for using TensorFlow V1 with GPU operations. You can find these steps [here.](https://www.tensorflow.org/install/gpu#older_versions_of_tensorflow)

Note, TensorFlow V1 is no longer being built for newer operating systems such as Ubuntu 20.04. 
Therefore, SparseML with TensorFlow V1 is unsupported on these operating systems as well.
