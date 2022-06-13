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
num_epochs: &num_epochs 135.0
init_lr: &init_lr 0.008

# Pruning Epoch/LR variables
pruning_recovery_start_epoch: &pruning_recovery_start_epoch 40.0
pruning_recovery_num_epochs: &pruning_recovery_num_epochs 60
pruning_recovery_start_lr: &pruning_recovery_start_lr 0.005
pruning_recovery_end_lr: &pruning_recovery_end_lr 0.00001

# Quantization Epoch/LR variables
quantization_start_epoch: &quantization_start_epoch 100
disable_quantization_observer_epoch: &disable_quantization_observer_epoch 115.0
freeze_bn_stats_epoch: &freeze_bn_stats_epoch 116.0
quantization_init_lr: &quantization_init_lr 0.01
quantization_lr_step_milestones: &quantization_lr_step_milestones [107, 114, 125]

# Block pruning management variables
pruning_start_epoch: &pruning_start_epoch 1.0
pruning_end_epoch: &pruning_end_epoch 36.0
pruning_update_frequency: &pruning_update_frequency 0.5
init_sparsity: &init_sparsity 0.05
mask_type: &mask_type [1, 4]

prune_low_target_sparsity: &prune_low_target_sparsity 0.7
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.75
prune_high_target_sparsity: &prune_high_target_sparsity 0.8
  
# Modifiers
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: *num_epochs

  # pruning phase
  - !SetLearningRateModifier
    start_epoch: 0.0
    learning_rate: *init_lr

  # pruning recovery phase
  - !LearningRateModifier
    start_epoch: *pruning_recovery_start_epoch
    end_epoch: *quantization_start_epoch
    lr_class: CosineAnnealingWarmRestarts
    lr_kwargs:
      lr_min: *pruning_recovery_end_lr
      cycle_epochs: *pruning_recovery_num_epochs
    init_lr: *pruning_recovery_start_lr

  - !SetWeightDecayModifier
    start_epoch: *pruning_recovery_start_epoch
    weight_decay: 0.0
    
  # quantization aware training phase
  - !LearningRateModifier
    start_epoch: *quantization_start_epoch
    lr_class: MultiStepLR
    lr_kwargs:
      milestones: *quantization_lr_step_milestones
      gamma: 0.1
    init_lr: *quantization_init_lr

pruning_modifiers:
  - !GMPruningModifier
    params:
      - sections.0.0.identity.conv.weight
      - sections.0.0.conv1.weight
      - sections.0.0.conv2.weight
      - sections.0.0.conv3.weight
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
    init_sparsity: *init_sparsity
    final_sparsity: *prune_low_target_sparsity
    mask_type: *mask_type
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
      - sections.2.3.conv1.weight
      - sections.2.3.conv3.weight
      - sections.2.4.conv1.weight
      - sections.2.4.conv3.weight
      - sections.2.5.conv1.weight
      - sections.2.5.conv3.weight
      - sections.3.1.conv1.weight
      - sections.3.2.conv1.weight
      - sections.1.0.conv2.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_mid_target_sparsity
    mask_type: *mask_type
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
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
    mask_type: *mask_type
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

quantization_modifiers:
  - !QuantizationModifier
    start_epoch: *quantization_start_epoch
    submodules:
      - input
      - sections
    disable_quantization_observer_epoch: *disable_quantization_observer_epoch
    freeze_bn_stats_epoch: *freeze_bn_stats_epoch

    # normally a ConstantPruningModifier and SetWeightDecayModifer=0 are also added here
    # but are unnecessary due to these already being set during the training phase
---

# ResNet-50 ImageNet Moderate Sparse Quantization

This recipe creates a sparse quantized [ResNet-50](https://arxiv.org/abs/1512.03385) model that achieves 99% of its baseline accuracy
on the ImageNet dataset.  Training was done using 4 GPUs using a total training batch size of 96
using an SGD optimizer.

When running, adjust hyperparameters based on training environment and dataset.

## Training

We can use the `sparseml.image_classification.train` utility for training.

*script command:*

```
python -m torch.distributed.launch \
  --nproc_per_node 4 \
  src/sparseml/pytorch/image_classification/train.py \
    --recipe-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate?recipe_type=original \
    --pretrained True \
    --arch-key resnet50 \
    --dataset imagenet \
    --dataset-path /PATH/TO/IMAGENET  \
    --train-batch-size 96 --test-batch-size 256 \
    --loader-num-workers 16 \
    --model-tag resnet50-imagenet-pruned_quant-moderate
```
