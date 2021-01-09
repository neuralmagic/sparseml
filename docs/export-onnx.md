## How to Export to ONNX

These are instructions for exporting to ONNX format for these popular frameworks.

### PyTorch ONNX
[ONNX support](https://pytorch.org/docs/stable/onnx.html) is natively built into PyTorch.
To enable ease of use, a high level API, `ModuleExporter`, is also included in the `sparseml.pytorch` package.
To run the export for a model, a sample batch must be provided. 
The sample batch is run through the model to freeze the execution graph into an ONNX format.

Example code:
```python
import os
import torch
from sparseml.pytorch.models import mnist_net
from sparseml.pytorch.utils import ModuleExporter

model = mnist_net()
export_dir = os.path.join(".", "mnist-export")
exporter = ModuleExporter(model, output_dir=export_dir)
print("exporting to onnx...")
exporter.export_onnx(sample_batch=torch.randn(1, 1, 28, 28))
print("exported onnx file in {}".format(export_dir))
```

### TensorFlow ONNX
ONNX support is not natively built into TensorFlow.
However, a third-party library, [tf2onnx](https://github.com/onnx/tensorflow-onnx), 
is maintained for the conversion to ONNX.
This pathway converts native protobuf graph definitions from TensorFlow into their equivalent ONNX representation.
Note, if you are using Python 3.5, then you will need to install tf2onnx version 1.5.6.

Example code:
```python
import os
from sparseml.tensorflow.utils import tf_compat, GraphExporter
from sparseml.tensorflow.models import mnist_net

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
        
        print("exporting to pb...")
        exporter.export_pb(outputs=[logits])
        print("exported pb file to {}".format(exporter.pb_path))

print("exporting to onnx...")
exporter.export_onnx(inputs=input_names, outputs=output_names)
print("exported onnx file to {}".format(exporter.onnx_path))