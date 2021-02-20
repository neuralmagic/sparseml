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

## Quick Tour

To enable flexibility, ease of use, and repeatability, optimizing a model is generally done using a recipe file.
The files encode the instructions needed for modifying the model and/or training process as a list of modifiers.
Example modifiers can be anything from setting the learning rate for the optimizer to gradual magnitude pruning.
The files are written in [YAML](https://yaml.org/) and stored in YAML or [markdown](https://www.markdownguide.org/) files using [YAML front matter](https://assemble.io/docs/YAML-front-matter.html).
The rest of the SparseML system is coded to parse the recipe files into a native format for the desired framework
and apply the modifications to the model and training pipeline.

A sample recipe for pruning a model generally looks like the following:

```yaml
version: 0.1.0
modifiers:
    - !EpochRangeModifier
        start_epoch: 0.0
        end_epoch: 70.0

    - !LearningRateModifier
        start_epoch: 0
        end_epoch: -1.0
        update_frequency: -1.0
        init_lr: 0.005
        lr_class: MultiStepLR
        lr_kwargs: {'milestones': [43, 60], 'gamma': 0.1}

    - !GMPruningModifier
        start_epoch: 0
        end_epoch: 40
        update_frequency: 1.0
        init_sparsity: 0.05
        final_sparsity: 0.85
        mask_type: unstructured
        params: ['sections.0.0.conv1.weight', 'sections.0.0.conv2.weight', 'sections.0.0.conv3.weight']
```

More information on the available recipes, formats, and arguments can be found [here](https://github.com/neuralmagic/sparseml/blob/main/docs/source/recipes.md). Additionally, all code implementations of the modifiers under the `optim` packages for the frameworks are documented with example YAML formats.

Pre-configured recipes and the resulting models can be explored and downloaded from the [SparseZoo](https://github.com/neuralmagic/sparsezoo). Also, [Sparsify](https://github.com/neuralmagic/sparsify) enables autoML style creation of optimization recipes for use with SparseML.

For a more in-depth read, check out [SparseML documentation](https://docs.neuralmagic.com/sparseml/).

### PyTorch Optimization

The PyTorch optimization libraries are located under the `sparseml.pytorch.optim` package.
Inside are APIs designed to make model optimization as easy as possible by integrating seamlessly into PyTorch training pipelines.

The integration is done using the `ScheduledOptimizer` class. It is intended to wrap your current optimizer and its step function. The step function then calls into the `ScheduledModifierManager` class which can be created from a recipe file. With this setup, the training process can then be modified as desired to optimize the model.

To enable all of this, the integration code you'll need to write is only a handful of lines:

```python
from sparseml.pytorch.optim import ScheduledModifierManager, ScheduledOptimizer

model = None  # your model definition
optimizer = None  # your optimizer definition
num_train_batches = len(train_data) / batch_size  # your number of batches per training epoch

manager = ScheduledModifierManager.from_yaml("/PATH/TO/recipe.yaml")
optimizer = ScheduledOptimizer(optimizer, model, manager, steps_per_epoch=num_train_batches)

# PyTorch training code...
```

### Keras Optimization

The Keras optimization libraries are located under the `sparseml.keras.optim` package.
Inside are APIs designed to make model optimization as easy as possible by integrating seamlessly into Keras training pipelines.

The integration is done using the `ScheduledModifierManager` class which can be created from a recipe file.
This class handles modifying the Keras objects for the desired optimizations using the `modify` method.
The edited model, optimizer, and any callbacks necessary to modify the training process are returned.
The model and optimizer can be used normally and the callbacks must be passed into the `fit` or `fit_generator` function.
If using `train_on_batch`, the callbacks must be invoked after each call.
After training is completed, call into the manager's `finalize` method to clean up the graph for exporting.

To enable all of this, the integration code you'll need to write is only a handful of lines:

```python
from sparseml.keras.optim import ScheduledModifierManager

model = None  # your model definition
optimizer = None  # your optimizer definition
num_train_batches = len(train_data) / batch_size  # your number of batches per training epoch

manager = ScheduledModifierManager.from_yaml("/PATH/TO/recipe.yaml")
model, optimizer, callbacks = manager.modify(
    model, optimizer, steps_per_epoch=num_train_batches
)

# Keras compilation and training code...
# Be sure to compile model after calling modify and pass the callbacks into the fit or fit_generator function.
# Note, if you are using train_on_batch, then you will need to invoke the callbacks after every step.
model.compile(...)
model.fit(..., callbacks=callbacks)

# finalize cleans up the graph for export
save_model = manager.finalize(model)
```

### TensorFlow V1 Optimization

The TensorFlow optimization libraries for TensorFlow version 1.X are located under the `sparseml.tensorflow_v1.optim` package. Inside are APIs designed to make model optimization as easy as possible by integrating seamlessly into TensorFlow V1 training pipelines.

The integration is done using the `ScheduledModifierManager` class which can be created from a recipe file.
This class handles modifying the TensorFlow graph for the desired optimizations.
With this setup, the training process can then be modified as desired to optimize the model.

#### Estimator-based pipelines

Estimator-based pipelines are simpler to integrate with as compared to session-based pipelines.
The `ScheduledModifierManager` can override the necessary callbacks in the estimator to modify the graph using the `modify_estimator` function.

```python
from sparseml.tensorflow_v1.optim import ScheduledModifierManager

estimator = None  # your estimator definition
num_train_batches = len(train_data) / batch_size  # your number of batches per training epoch

manager = ScheduledModifierManager.from_yaml("/PATH/TO/config.yaml")
manager.modify_estimator(estimator, steps_per_epoch=num_train_batches)

# Normal estimator training code...
```

#### Session-based pipelines

Session-based pipelines need a little bit more as compared to estimator-based pipelines; however,
it is still designed to require only a few lines of code for integration.
After graph creation, the manager's `create_ops` method must be called.
This will modify the graph as needed for the optimizations and return modifying ops and extras.
After creating the session and training normally, call into `session.run` with the modifying ops after each step.
Modifying extras contain objects such as tensorboard summaries of the modifiers to be used if desired.
Finally, once completed, `complete_graph` must be called to remove the modifying ops for saving and export.

```python
from sparseml.tensorflow_v1.utils import tf_compat
from sparseml.tensorflow_v1.optim import ScheduledModifierManager


with tf_compat.Graph().as_default() as graph:
    # Normal graph setup....
    num_train_batches = len(train_data) / batch_size  # your number of batches per training epoch
    
    # Modifying graphs, be sure his is called after graph is created and before session is created
    manager = ScheduledModifierManager.from_yaml("/PATH/TO/config.yaml")
    mod_ops, mod_extras = manager.create_ops(steps_per_epoch=num_train_batches)
    
    with tf_compat.Session() as sess:
        # Normal training code...
        # Call sess.run with the mod_ops after every batch update
        sess.run(mod_ops)
    
        # Call into complete_graph after training is done
        manager.complete_graph()
```

### Exporting to ONNX

[ONNX](https://onnx.ai/) is a generic representation for neural network graphs that most ML frameworks can be converted to. Some inference engines such as [DeepSparse](https://github.com/neuralmagic/deepsparse) natively take in ONNX for deployment pipelines, so convenience functions for conversion and export are provided for the supported frameworks.

#### Exporting PyTorch to ONNX

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

#### Exporting Keras to ONNX

ONNX is not built into the Keras system, but is supported through an ONNX official tool [keras2onnx](https://github.com/onnx/keras-onnx). The `ModelExporter` class under the `sparseml.keras.utils` package features an `export_onnx` function built on top of keras2onnx.
Example code:

```python
import os
from sparseml.keras.utils import ModelExporter

model = None  # fill in with your model
exporter = ModelExporter(model, output_dir=os.path.join(".", "onnx-export"))
exporter.export_onnx()
```

#### Exporting TensorFlow V1 to ONNX

ONNX is not built into the TensorFlow system, but it is supported through an ONNX official tool
[tf2onnx](https://github.com/onnx/tensorflow-onnx).
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
