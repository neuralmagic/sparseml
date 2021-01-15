# Recalibrating in PyTorch

[TODO: ENGINEERING: EDIT THE FOLLOWING SO IT REFLECTS ANY UPDATES; THEN REMOVE THIS COMMENT.]

The recalibration tooling for PyTorch is located under `neuralmagicML.pytorch.recal`.
Inside are APIs designed to make model recalibration as easy as possible.
Additionally, the tooling is designed to work with the previously described [config files](docs/recal-config.md).

- `ScheduledModifierManager`: creates modifiers from a config file. Specifically, `ScheduledModifierManager.from_yaml(/PATH/TO/config.yaml)` should be used.

- The function call will return a new instance of `ScheduledModifierManager` containing the modifiers described in the config file.

- Once a manager class has been created, a `ScheduledOptimizer` class must be created.
This class is used to wrap the `ScheduledModifierManager`, PyTorch model, and PyTorch optimizer to enable modifying the training process.

- The `ScheduledOptimizer` should then be used in place of the original PyTorch optimizer in the rest of your code.
Mainly, it overrides the `optimizer.step()` function to modify the training process.

- Additionally, `optimizer.epoch_start()` and `optimizer.epoch_end()` should be called 
at the start and end of each epoch, respectively. 

Example:

```python
import math
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as TF
from neuralmagicML.pytorch.models import mnist_net
from neuralmagicML.pytorch.datasets import MNISTDataset
from neuralmagicML.pytorch.recal import ScheduledModifierManager, ScheduledOptimizer

model = mnist_net()
optimizer = Adam(model.parameters(), lr=1e-4)
train_data = MNISTDataset(train=True)
batch_size = 1024

config_path = "/PATH/TO/config.yaml"
manager = ScheduledModifierManager.from_yaml(config_path)
optimizer = ScheduledOptimizer(optimizer, model, manager, steps_per_epoch=math.ceil(len(train_data) / batch_size))

for epoch in range(manager.max_epochs):
    for batch_x, batch_y in DataLoader(train_data, batch_size):
        optimizer.zero_grad()
        batch_pred = model(batch_x)[0]
        loss = TF.cross_entropy(batch_pred, batch_y)
        loss.backward()
        optimizer.step()
```

Note: if you would like to log to TensorBoard, a logger class can be created and passed into the `ScheduledOptimizer`.
Example:

```python
from neuralmagicML.pytorch.utils import TensorBoardLogger

optimizer = ScheduledOptimizer(
    optimizer, 
    model, 
    manager, 
    steps_per_epoch=math.ceil(len(train_data) / batch_size), 
    loggers=[TensorBoardLogger()]
)
```
