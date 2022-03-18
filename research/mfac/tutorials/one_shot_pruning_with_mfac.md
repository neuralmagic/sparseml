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

# One-Shot Pruning with M-FAC

This tutorial shows how to one-shot prune an example MNISTNet model using
both magnitude and [Matrix-Free Approximate Curvature (M-FAC)](https://arxiv.org/pdf/2107.03356.pdf)
pruning.
 
## Overview
Magnitude pruning selects weights with the smallest magnitudes for removal from a model.
This technique provides a strong standard for model pruning that can prune models to reasonably
high sparsities with minimal accuracy degradation.
Instead of magnitudes, the M-FAC pruning algorithm efficiently uses first-order model
information (gradients) to approximate second-order information that can be used to
decide which weights are optimal to prune. This technique can often lead to superior
results to magnitude pruning when pruning in both a one-shot and gradual setting.

To demonstrate the benefits of M-FAC pruning in a one-shot scenario, in this tutorial you will
prune a small CNN trained on the MNIST dataset. You will prune the model to 35% sparsity using
both M-FAC and magnitude pruning. In the process, you will learn how to perform the one-shot
pruning by using a recipe as well as be able to compare the results.

### Steps
1. Setting Up
2. Inspecting the Recipes
3. Applying Recipes in One Shot
3. One-Shot Pruning with M-FAC
4. One-Shot Pruning with Magnitude Pruning
5. Comparing the Results

## Need Help?

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)

## Setting Up

This tutorial can be run by cloning and installing the `sparseml` repository which contains scripts and recipes for
this pruning example:

```bash
git clone https://github.com/neuralmagic/sparseml.git
pip install sparseml[torchvision]
cd sparseml
```

## Inspecting the Recipes

To run both M-FAC and magnitude one-shot pruning, you will need two separate recipes.
For this tutorial, the recipes for pruning MNISTNet to 35% sparsity are provided
in SparseML. They can be viewed with the command line or by following the links below:

* M-FAC: [research/mfac/recipes/pruning-mnistnet-one_shot-mfac.md](https://github.com/neuralmagic/sparseml/blob/main/research/mfac/recipes/pruning-mnistnet-one_shot-mfac.md)
* Magnitude: [research/mfac/recipes/pruning-mnistnet-one_shot-magnitude.md](https://github.com/neuralmagic/sparseml/blob/main/research/mfac/recipes/pruning-mnistnet-one_shot-magnitude.md)

Notice that the recipes are identical except that the `MFACPruningModifier` and `mfac_options`
are used in M-FAC pruning.

The remainder of this tutorial can be run in a single Python REPL
(launch by entering `python` in your command prompt).

## Applying Recipes in One Shot
All SparseML recipes can be applied to a model in one shot by using
`ScheduledModifierManger.apply`. The given model will automatically
be updated to reflect the applied final state of the recipe.

### Loading Pre-Trained Models

Before applying the recipes, we must first load our pre-trained models.
SparseML helper functions can be used for this.  You will load two copies of the
same model to be used for M-FAC and Magnitude pruning.

```python
import torch
from sparseml.pytorch.models import ModelRegistry

device = "cuda" if torch.cuda.is_available() else "cpu"

model_mfac = ModelRegistry.create("mnistnet", pretrained=True).to(device)
model_magnitude = ModelRegistry.create("mnistnet", pretrained=True).to(device)
```

## One-Shot Pruning with M-FAC
When applying M-FAC pruning in one shot, the M-FAC algorithm must
still be able to obtain first-order information from the model in order to determine
which weights to select for pruning.  To do this, a `GradSampler` object must
be provided to the `ScheudledModifierManager.apply` class.

### Creating a GradSampler

A `GradSampler` defines a data loader and loss function for the M-FAC algorithm to use
to extract gradients from the model to prune with. The code block below
creates a `GradSampler` that has a data loader that loads the MNIST dataset to the correct
device at batch size 16 and has a cross entropy loss function.

The data loader must yield tuples of 
`(forward_args: List, forward_kwargs: Dict, loss_targets: Any)`
so that the forward and backwards passes may be run with the loss function as:
```python
forward_args, forward_kwargs, loss_targets = next(data_loader)

forward_outputs = module(*forward_args, **forward_kwargs)
loss = loss_fn(forward_outputs, loss_targets)
loss.backwards()
```

Run the code below to create the grad sampler:

```python
from torch.utils.data import DataLoader
from sparseml.pytorch.datasets import MNISTDataset
from sparseml.pytorch.utils import GradSampler

# create base data loader
dataset = MNISTDataset(train=True)
data_loader = DataLoader(dataset, shuffle=True, batch_size=16)

# wrap data loader to send data to correct device 
def device_data_loader():
    for sample in data_loader:
        img, target = [t.to(device) if isinstance(t, torch.Tensor) else t for t in sample]
        yield [img], {}, target

# wrap cross entropy loss function to use correct output
def loss_function(model_outputs, loss_target):
    return torch.nn.functional.cross_entropy(model_outputs[0], loss_target)

# create grad sampler with the created data laoder and loss function
grad_sampler = GradSampler(device_data_loader(), loss_function)
```

### Applying the M-FAC Recipe
Now, you are ready to apply the recipe in one shot using the GradSampler.
Using the given recipe, the GradSampler will be used by the manager to collect 512 gradient
samples from the model before applying M-FAC pruning to prune the model to 35% sparsity.

```python
from sparseml.pytorch.optim import ScheduledModifierManager

manager_mfac = ScheduledModifierManager.from_yaml("research/mfac/recipes/pruning-mnistnet-one_shot-mfac.md")
manager_mfac.apply(model_mfac, grad_sampler=grad_sampler)
```

## One-Shot Pruning with Magnitude Pruning
Magnitude pruning recipes just look at the pre-trained model weights, so extra
information is not needed to prune the model in one shot. The recipe can be applied directly
to prune the model to 35% sparsity:

```python
from sparseml.pytorch.optim import ScheduledModifierManager

manager_magnitude = ScheduledModifierManager.from_yaml("research/mfac/recipes/pruning-mnistnet-one_shot-magnitude.md")
manager_magnitude.apply(model_magnitude)
```


## Comparing the Results
Especially in one-shot pruning, there tends to be a large difference between the results of
M-FAC and magnitude pruning.  To view this, you will evaluate both the M-FAC and magnitude
pruned models against the MNIST dataset.  The following code prints the accuracies of both models
after running them through respectively.

```python
from tqdm.auto import tqdm

# load validation dataset
val_dataset = MNISTDataset(train=False)

# define evaluation function
def eval_mnist(model, method):
    print(f"Evaluating {method} pruned MNIST model...")
    model.eval()
    num_correct = 0
    for img, target in tqdm(val_dataset):
        img = img.unsqueeze(0).to(device)
        pred = torch.argmax(model(img)[1])
        if pred.item() == target:
            num_correct += 1
    print(f"MNIST Val Accuracy {method}: {num_correct / len(val_dataset)}")

# evaluate the M-FAC and magnitude pruned models
eval_mnist(model_mfac, "M-FAC")
eval_mnist(model_magnitude, "Magnitude")
```

After running the above evaluation code, you should see that M-FAC pruning greatly outperforms
magnitude pruning:

```
MNIST Val Accuracy M-FAC: 0.9798
MNIST Val Accuracy Magnitude: 0.4861
```

The M-FAC pruned model drops less than 2% from the baseline accuracy of 99% to 97.9% whereas
the magnitude pruned model drops to 48.6% accuracy.

## Wrap-Up
In this tutorial you applied both M-FAC and magnitude pruning in one shot and compared
their results. More information about M-FAC pruning and other tutorials can be found
[here](https://github.com/neuralmagic/sparseml/blob/main/research/mfac).

For Neural Magic Support, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)
