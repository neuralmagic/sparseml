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

# ONNX Export

[ONNX](https://onnx.ai/) is a generic representation for neural network graphs that most ML frameworks can be converted to. Some inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse) natively take in ONNX for deployment pipelines, so convenience functions for conversion and export are provided for the supported frameworks.

## Exporting PyTorch to ONNX

ONNX is built into the PyTorch system natively.
The `ModuleExporter` class under the `sparseml.pytorch.utils` package features an `export_onnx` function built on top of this native support.
Example code:

```python
import os
import torch
from sparseml.pytorch.models import mnist_net
from sparseml.pytorch.utils import ModuleExporter

model = mnist_net()
exporter = ModuleExporter(model, output_dir=os.path.join(".", "onnx-export"))
exporter.export_onnx(sample_batch=torch.randn(1, 1, 28, 28))
```

## Exporting Keras to ONNX

ONNX is not built into the Keras system, but is supported through an ONNX official tool [keras2onnx.](https://github.com/onnx/keras-onnx) The `ModelExporter` class under the `sparseml.keras.utils` package features an `export_onnx` function built on top of keras2onnx.
Example code:

```python
import os
from sparseml.keras.utils import ModelExporter

model = None  # fill in with your model
exporter = ModelExporter(model, output_dir=os.path.join(".", "onnx-export"))
exporter.export_onnx()
```

## Exporting TensorFlow V1 to ONNX

ONNX is not built into the TensorFlow system, but it is supported through an ONNX official tool
[tf2onnx.](https://github.com/onnx/tensorflow-onnx)
The `GraphExporter` class under the `sparseml.tensorflow_v1.utils` package features an
`export_onnx` function built on top of tf2onnx.
Note that the ONNX file is created from the protobuf graph representation, so `export_pb` must be called first.
Example code:

```python
import os
from sparseml.tensorflow_v1.utils import tf_compat, GraphExporter
from sparseml.tensorflow_v1.models import mnist_net

exporter = GraphExporter(output_dir=os.path.join(".", "mnist-tf-export"))

with tf_compat.Graph().as_default() as graph:
    inputs = tf_compat.placeholder(
        tf_compat.float32, [None, 28, 28, 1], name="inputs"
    )
    logits = mnist_net(inputs)
    input_names = [inputs.name]
    output_names = [logits.name]

    with tf_compat.Session() as sess:
        sess.run(tf_compat.global_variables_initializer())
        exporter.export_pb(outputs=[logits])

exporter.export_onnx(inputs=input_names, outputs=output_names)
```
