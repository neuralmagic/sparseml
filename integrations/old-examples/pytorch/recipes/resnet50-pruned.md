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
num_epochs: &num_epochs 90
init_lr: &init_lr 0.0028
lr_step_epochs: &lr_step_epochs [43, 60]

# Pruning variables
pruning_start_epoch: &pruning_start_epoch 0.0
pruning_end_epoch: &pruning_end_epoch 40.0
pruning_update_frequency: &pruning_update_frequency 0.4
init_sparsity: &init_sparsity 0.05
final_sparsity: &final_sparsity 0.88
pruning_mask_type: &pruning_mask_type unstructured

# Modifiers
training_modifiers:
  - !EpochRangeModifier
    end_epoch: *num_epochs
    start_epoch: 0.0

  - !LearningRateModifier
    constant_logging: False
    end_epoch: -1.0
    init_lr: *init_lr
    lr_class: MultiStepLR
    lr_kwargs: {'milestones': *lr_step_epochs, 'gamma': 0.1}
    start_epoch: 0.0
    update_frequency: -1.0
        
pruning_modifiers:
  - !GMPruningModifier
    end_epoch: *pruning_end_epoch
    final_sparsity: *final_sparsity
    init_sparsity: *init_sparsity
    inter_func: cubic
    leave_enabled: True
    mask_type: *pruning_mask_type
    params: ['sections.0.0.conv1.weight', 'sections.0.0.conv2.weight', 'sections.0.0.conv3.weight', 'sections.0.0.identity.conv.weight', 'sections.0.1.conv1.weight', 'sections.0.1.conv2.weight', 'sections.0.1.conv3.weight', 'sections.0.2.conv1.weight', 'sections.0.2.conv2.weight', 'sections.0.2.conv3.weight', 'sections.1.0.conv1.weight', 'sections.1.0.conv2.weight', 'sections.1.0.conv3.weight', 'sections.1.0.identity.conv.weight', 'sections.1.1.conv1.weight', 'sections.1.1.conv2.weight', 'sections.1.1.conv3.weight', 'sections.1.2.conv1.weight', 'sections.1.2.conv2.weight', 'sections.1.2.conv3.weight', 'sections.1.3.conv1.weight', 'sections.1.3.conv2.weight', 'sections.1.3.conv3.weight', 'sections.2.0.conv1.weight', 'sections.2.0.conv2.weight', 'sections.2.0.conv3.weight', 'sections.2.0.identity.conv.weight', 'sections.2.1.conv1.weight', 'sections.2.1.conv2.weight', 'sections.2.1.conv3.weight', 'sections.2.2.conv1.weight', 'sections.2.2.conv2.weight', 'sections.2.2.conv3.weight', 'sections.2.3.conv1.weight', 'sections.2.3.conv2.weight', 'sections.2.3.conv3.weight', 'sections.2.4.conv1.weight', 'sections.2.4.conv2.weight', 'sections.2.4.conv3.weight', 'sections.2.5.conv1.weight', 'sections.2.5.conv2.weight', 'sections.2.5.conv3.weight', 'sections.3.0.conv1.weight', 'sections.3.0.conv2.weight', 'sections.3.0.conv3.weight', 'sections.3.0.identity.conv.weight', 'sections.3.1.conv1.weight', 'sections.3.1.conv2.weight', 'sections.3.1.conv3.weight', 'sections.3.2.conv1.weight', 'sections.3.2.conv2.weight', 'sections.3.2.conv3.weight']
    start_epoch: *pruning_start_epoch
    update_frequency: *pruning_update_frequency
---

# ResNet-50 ImageNet Moderate Sparse

This recipe creates a sparse [ResNet-50](https://arxiv.org/abs/1512.03385) model that
achieves 99% recovery of its baseline accuracy on the ImageNet dataset.
Training was done using 4 GPUs using a total training batch size of 1024
using an SGD optimizer.

When running, adjust hyperparameters based on training environment and dataset.

## Training
We can use the `sparseml.image_classification.train` utility for training.

*script command:*

```
sparseml.image_classification.train \
    --recipe-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate?recipe_type=original \
    --pretrained True \
    --arch-key resnet50 \
    --dataset imagenet \
    --dataset-path /PATH/TO/IMAGENET  \
    --train-batch-size 1024 --test-batch-size 2056 \
    --loader-num-workers 16 \
    --model-tag resnet50-imagenet-pruned-moderate
```
