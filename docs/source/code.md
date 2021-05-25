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

# Sparsification Code

## PyTorch Sparsification

The PyTorch sparsification libraries are located under the `sparseml.pytorch.optim` package.
Inside are APIs designed to make model sparsification as easy as possible by integrating seamlessly into PyTorch training pipelines.

The step function then calls into the `ScheduledModifierManager` class which can be created from a recipe file.
The `modify()` function wraps an optimizer or optimizer like object (contains a step function) to override the step invocation.
With this setup, the training process can then be modified as desired to sparsify the model.

To enable all of this, the integration code is accomplished by writing a handful of lines:

```python
from sparseml.pytorch.optim import ScheduledModifierManager

model = Model()  # model definition
optimizer = Optimizer()  # optimizer definition
train_data = TrainData()  # train data definition
batch_size = BATCH_SIZE  # training batch size
steps_per_epoch = len(train_data) // batch_size

manager = ScheduledModifierManager.from_yaml(PATH_TO_RECIPE)
optimizer = manager.modify(model, optimizer, steps_per_epoch)

# PyTorch training code

manager.finalize(model)
```

## Keras Sparsification

The Keras sparsification libraries are located under the `sparseml.keras.optim` package.
Inside are APIs designed to make model sparsification as easy as possible by integrating seamlessly into Keras training pipelines.

The integration is done using the `ScheduledModifierManager` class which can be created from a recipe file.
This class handles modifying the Keras objects for the desired algorithms using the `modify` method.
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

## TensorFlow V1 Sparsification

The TensorFlow sparsification libraries for TensorFlow version 1.X are located under the `sparseml.tensorflow_v1.optim` package. 
Inside are APIs designed to make model sparsification as easy as possible by integrating seamlessly into TensorFlow V1 training pipelines.

The integration is done using the `ScheduledModifierManager` class which can be created from a recipe file.
This class handles modifying the TensorFlow graph for the desired algorithms.
With this setup, the training process can then be modified as desired to sparsify the model.

### Estimator-Based Pipelines

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

### Session-Based Pipelines

Session-based pipelines need a little bit more as compared to estimator-based pipelines; however,
it is still designed to require only a few lines of code for integration.
After graph creation, the manager's `create_ops` method must be called.
This will modify the graph as needed for the algorithms and return modifying ops and extras.
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
