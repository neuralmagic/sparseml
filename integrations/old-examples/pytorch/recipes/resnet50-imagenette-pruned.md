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

---
# General Epoch/LR variables
num_epochs: &num_epochs 10
lr: &lr 0.008

# Pruning variables
pruning_start_epoch: &pruning_start_epoch 1.0
pruning_end_epoch: &pruning_end_epoch 8.0
pruning_update_frequency: &pruning_update_frequency 0.5
init_sparsity: &init_sparsity 0.05

prune_low_target_sparsity: &prune_low_target_sparsity 0.8
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.85
prune_high_target_sparsity: &prune_high_target_sparsity 0.9

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: *num_epochs

  - !SetLearningRateModifier
    start_epoch: 0.0
    learning_rate: *lr

pruning_modifiers:
  - !GMPruningModifier
    params:
      - sections.0.0.conv1.weight
      - sections.0.0.conv2.weight
      - sections.0.0.conv3.weight
      - sections.0.0.identity.conv.weight
      - sections.0.1.conv1.weight
      - sections.0.1.conv3.weight
      - sections.0.2.conv1.weight
      - sections.0.2.conv3.weight
      - sections.1.0.conv1.weight
      - sections.1.0.conv3.weight
      - sections.1.2.conv3.weight
      - sections.1.3.conv1.weight
      - sections.2.0.conv1.weight
      - sections.3.0.conv1.weight
      - classifier.fc.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_low_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
      - sections.0.1.conv2.weight
      - sections.0.2.conv2.weight
      - sections.1.0.identity.conv.weight
      - sections.1.1.conv1.weight
      - sections.1.1.conv2.weight
      - sections.1.1.conv3.weight
      - sections.1.2.conv1.weight
      - sections.1.2.conv2.weight
      - sections.1.3.conv2.weight
      - sections.1.3.conv3.weight
      - sections.2.0.conv3.weight
      - sections.2.0.identity.conv.weight
      - sections.2.1.conv1.weight
      - sections.2.1.conv3.weight
      - sections.2.2.conv1.weight
      - sections.2.2.conv3.weight
      - sections.2.3.conv1.weight
      - sections.2.3.conv3.weight
      - sections.2.4.conv1.weight
      - sections.2.4.conv3.weight
      - sections.2.5.conv1.weight
      - sections.2.5.conv3.weight
      - sections.3.1.conv1.weight
      - sections.3.2.conv1.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_mid_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
      - sections.1.0.conv2.weight
      - sections.2.0.conv2.weight
      - sections.2.1.conv2.weight
      - sections.2.2.conv2.weight
      - sections.2.3.conv2.weight
      - sections.2.4.conv2.weight
      - sections.2.5.conv2.weight
      - sections.3.0.conv2.weight
      - sections.3.0.conv3.weight
      - sections.3.0.identity.conv.weight
      - sections.3.1.conv2.weight
      - sections.3.1.conv3.weight
      - sections.3.2.conv2.weight
      - sections.3.2.conv3.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_high_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
---

# ResNet-50 Imagenette Conservative Sparse

This recipe creates a sparse [ResNet-50](https://arxiv.org/abs/1512.03385) model that
achieves full recovery of its baseline accuracy on the Imagenette dataset.
Training was done using 1 GPU with a total training batch size of 128
using a SGD optimizer.

When running, adjust hyperparameters based on training environment and dataset.

## Training
We can use the `sparseml.image_classification.train` utility for training. 
Alternatively, a full walk-through notebook is located at `sparseml/integrations/pytorch/notebooks/classification.ipynb`.

*script command:*

```
sparseml.image_classification.train \
    --recipe-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenette/pruned-conservative?recipe_type=original \
    --pretrained True \
    --arch-key resnet50 \
    --dataset imagenette \
    --dataset-path /PATH/TO/IMAGENETTE  \
    --train-batch-size 128 --test-batch-size 256 \
    --loader-num-workers 8 \
    --model-tag resnet50-imagenette-pruned-conservative
```
