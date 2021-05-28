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
num_epochs: &num_epochs 242
init_lr: &init_lr 0.0032

# Pruning Epoch/LR variables
pruning_recovery_start_lr: &pruning_recovery_start_lr 0.0032
pruning_recovery_lr_step_size: &pruning_recovery_lr_step_size 1
pruning_recovery_lr_gamma: &pruning_recovery_lr_gamma 0.985

# Quantization Epoch/LR variables
quantization_start_epoch: &quantization_start_epoch 240
quantization_init_lr: &quantization_init_lr 0.0032

# Block pruning management variables
pruning_start_epoch: &pruning_start_epoch 0
pruning_end_epoch: &pruning_end_epoch 100
pruning_update_frequency: &pruning_update_frequency 0.5
init_sparsity: &init_sparsity 0.05
mask_type: &mask_type [1, 4]

prune_none_target_sparsity: &prune_none_target_sparsity 0.4
prune_low_target_sparsity: &prune_low_target_sparsity 0.75
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.8
prune_high_target_sparsity: &prune_high_target_sparsity 0.85
  
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
    constant_logging: False
    end_epoch: *quantization_start_epoch
    init_lr: *pruning_recovery_start_lr
    log_types: __ALL__
    lr_class: StepLR
    lr_kwargs: {
        'step_size': *pruning_recovery_lr_step_size,
        'gamma': *pruning_recovery_lr_gamma
      }
    start_epoch: *pruning_end_epoch
    update_frequency: -1.0
    
  - !SetWeightDecayModifier
    start_epoch: *pruning_end_epoch
    weight_decay: 0.0
    
  # quantization aware training phase
  - !SetLearningRateModifier
    start_epoch: *quantization_start_epoch
    learning_rate: *quantization_init_lr

pruning_modifiers:
  - !GMPruningModifier
    params:
      - model.16.conv.weight
      - model.19.cv1.conv.weight
      - model.26.cv2.conv.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_none_target_sparsity
    mask_type: *mask_type
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
      - model.12.cv1.conv.weight
      - model.14.conv.weight
      - model.2.cv1.conv.weight
      - model.20.cv2.conv.weight
      - model.22.conv.weight
      - model.26.cv1.conv.weight
      - model.27.1.cv1.conv.weight
      - model.28.m.0.weight
      - model.28.m.2.weight
      - model.4.0.cv1.conv.weight
      - model.6.1.cv1.conv.weight
      - model.6.2.cv1.conv.weight
      - model.6.3.cv1.conv.weight
      - model.6.4.cv1.conv.weight
      - model.6.5.cv1.conv.weight
      - model.6.6.cv1.conv.weight
      - model.6.7.cv1.conv.weight
      - model.8.0.cv1.conv.weight
      - model.8.1.cv1.conv.weight
      - model.8.2.cv1.conv.weight
      - model.8.3.cv1.conv.weight
      - model.8.4.cv1.conv.weight
      - model.8.5.cv1.conv.weight
      - model.8.6.cv1.conv.weight
      - model.8.7.cv1.conv.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_low_target_sparsity
    mask_type: *mask_type
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
      - model.1.conv.weight
      - model.10.0.cv1.conv.weight
      - model.10.1.cv1.conv.weight
      - model.10.2.cv1.conv.weight
      - model.10.3.cv1.conv.weight
      - model.11.cv1.conv.weight
      - model.12.cv2.conv.weight
      - model.19.cv2.conv.weight
      - model.2.cv2.conv.weight
      - model.27.0.cv1.conv.weight
      - model.27.0.cv2.conv.weight
      - model.27.1.cv2.conv.weight
      - model.28.m.1.weight
      - model.3.conv.weight
      - model.4.0.cv2.conv.weight
      - model.4.1.cv1.conv.weight
      - model.4.1.cv2.conv.weight
      - model.5.conv.weight
      - model.6.0.cv1.conv.weight
      - model.6.0.cv2.conv.weight
      - model.6.1.cv2.conv.weight
      - model.6.2.cv2.conv.weight
      - model.6.3.cv2.conv.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_mid_target_sparsity
    mask_type: *mask_type
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency

  - !GMPruningModifier
    params:
      - model.10.0.cv2.conv.weight
      - model.10.1.cv2.conv.weight
      - model.10.2.cv2.conv.weight
      - model.10.3.cv2.conv.weight
      - model.11.cv2.conv.weight
      - model.13.conv.weight
      - model.15.conv.weight
      - model.20.cv1.conv.weight
      - model.21.conv.weight
      - model.23.conv.weight
      - model.6.4.cv2.conv.weight
      - model.6.5.cv2.conv.weight
      - model.6.6.cv2.conv.weight
      - model.6.7.cv2.conv.weight
      - model.7.conv.weight
      - model.8.0.cv2.conv.weight
      - model.8.1.cv2.conv.weight
      - model.8.2.cv2.conv.weight
      - model.8.3.cv2.conv.weight
      - model.8.4.cv2.conv.weight
      - model.8.5.cv2.conv.weight
      - model.8.6.cv2.conv.weight
      - model.8.7.cv2.conv.weight
      - model.9.conv.weight
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
      - model.0
      - model.1 
      - model.2
      - model.3 
      - model.4
      - model.5
      - model.6
      - model.7
      - model.8
      - model.9
      - model.10
      - model.11
      - model.12
      - model.13
      - model.14
      - model.15
      - model.16
      - model.17
      - model.18
      - model.19
      - model.20
      - model.21
      - model.22
      - model.23
      - model.24
      - model.25
      - model.26
      - model.27
---

# YOLOv3-SPP Pruned-Quantized

This recipe creates a sparse-quantized, [YOLOv3-SPP](https://arxiv.org/abs/1804.02767) model that achieves 94% recovery of its baseline accuracy on the COCO detection dataset.
Training was done using 4 GPUs at half precision using a total training batch size of 256 with the
[SparseML integration with ultralytics/yolov3](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov3).

When running, adjust hyperparameters based on training environment and dataset.

Note that half-precision, EMA, and pickling are not supported for quantization.
Therefore, once quantization is run, all three will be disabled for the training pipeline.
This additionally means that the checkpoints are saved using state_dicts rather than pickels of the model.

## Weights and Biases

- [YOLOv3-SPP LeakyReLU on VOC](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/2dfy3rgs)

## Training

To set up the training environment, follow the instructions on the [integration README](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/README.md).
Using the given training script from the `yolov3` directory the following command can be used to launch this recipe.  
The contents of the `hyp.pruned_quantized.yaml` hyperparameters file is given below.
Adjust the script command for your GPU device setup. 
Ultralytics supports both DataParallel and DDP.

*script command:*

```
python train.py \
    --recipe ../recipes/yolov3-spp.pruned.md \
    --weights PRETRAINED_WEIGHTS \
    --data voc.yaml \
    --hyp ../data/hyp.pruned_quantized.yaml \
    --name yolov3-spp-lrelu-pruned-quantized
```

hyp.prune_quantized.yaml:
```yaml
lr0: 0.0
lrf: 0.0
momentum: 0.843
weight_decay: 0.00036
warmup_epochs: 40.0
warmup_momentum: 0.5
warmup_bias_lr: 0.05
box: 0.0296
cls: 0.243
cls_pw: 0.631
obj: 0.301
obj_pw: 0.911
iou_t: 0.2
anchor_t: 2.91
fl_gamma: 0.0
hsv_h: 0.0138
hsv_s: 0.664
hsv_v: 0.464
degrees: 0.373
translate: 0.245
scale: 0.898
shear: 0.602
perspective: 0.0
flipud: 0.00856
fliplr: 0.5
mosaic: 1.0
mixup: 0.243
```