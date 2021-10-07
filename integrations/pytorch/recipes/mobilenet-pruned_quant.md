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
num_epochs: &num_epochs 100.0
init_lr: &init_lr 0.005

# Pruning Epoch/LR variables
pruning_recovery_start_epoch: &pruning_recovery_start_epoch 40.0
pruning_recovery_num_epochs: &pruning_recovery_num_epochs 60
pruning_recovery_start_lr: &pruning_recovery_start_lr 0.005

# Quantization Epoch/LR variables
quantization_start_epoch: &quantization_start_epoch 90
disable_quantization_observer_epoch: &disable_quantization_observer_epoch 92.0
freeze_bn_stats_epoch: &freeze_bn_stats_epoch 92.0
quantization_init_lr: &quantization_init_lr 0.00001
quantization_lr_step_milestones: &quantization_lr_step_milestones [91]

# Block pruning management variables
pruning_start_epoch: &pruning_start_epoch 1.0
pruning_end_epoch: &pruning_end_epoch 36.0
pruning_update_frequency: &pruning_update_frequency 0.5
init_sparsity: &init_sparsity 0.05
mask_type: &mask_type [1, 4]

prune_low_target_sparsity: &prune_low_target_sparsity 0.55
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.7
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
      cycle_epochs: *quantization_start_epoch
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
      - sections.0.0.point.conv.weight
      - sections.1.0.point.conv.weight
      - sections.1.1.point.conv.weight
      - sections.2.0.point.conv.weight
      - classifier.fc.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_low_target_sparsity
    mask_type: *mask_type
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
      - sections.2.1.point.conv.weight
      - sections.3.0.point.conv.weight
      - sections.3.1.point.conv.weight
      - sections.3.5.point.conv.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_mid_target_sparsity
    mask_type: *mask_type
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
      - sections.3.2.point.conv.weight
      - sections.3.3.point.conv.weight
      - sections.3.4.point.conv.weight
      - sections.4.0.point.conv.weight
      - sections.4.1.point.conv.weight
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

# MobileNet-V1 ImageNet Moderate Sparse Quantization

This recipe creates a sparse quantized [MobileNet-V1](https://arxiv.org/abs/1704.04861) model
that achieves 99% of its baseline accuracy on the ImageNet dataset.  Training was done using
4 GPUs using a total training batch size of 96 using an SGD optimizer.

When running, adjust hyperparameters based on training environment and dataset.

## Training

The training script can be found at `sparseml/integrations/pytorch/train.py`.

*script command:*

```
python -m torch.distributed.launch \
  --nproc_per_node 4 \
  integrations/pytorch/train.py \
    --recipe-path zoo:cv/classification/mobilenet_v1-1.0/pytorch-sparseml/imagenet/pruned_quant-moderate?recipe_type=original \
    --pretrained True \
    --arch-key mobilenet \
    --dataset imagenet \
    --dataset-path /PATH/TO/IMAGENET  \
    --train-batch-size 96 --test-batch-size 256 \
    --loader-num-workers 16 \
    --model-tag mobilenet-imagenet-pruned_quant-moderate
```
