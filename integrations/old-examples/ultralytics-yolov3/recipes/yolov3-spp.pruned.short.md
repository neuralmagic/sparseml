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
num_epochs: &num_epochs 80

# pruning hyperparameters
init_sparsity: &init_sparsity 0.05
pruning_start_epoch: &pruning_start_epoch 0
pruning_end_epoch: &pruning_end_epoch 40
update_frequency: &pruning_update_frequency 0.5

prune_none_target_sparsity: &prune_none_target_sparsity 0.4
prune_low_target_sparsity: &prune_low_target_sparsity 0.75
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.8
prune_high_target_sparsity: &prune_high_target_sparsity 0.85

# modifiers
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: *num_epochs

pruning_modifiers:
  - !GMPruningModifier
    params:
      - model.16.conv.weight
      - model.19.cv1.conv.weight
      - model.26.cv2.conv.weight
    init_sparsity: *init_sparsity
    final_sparsity: *prune_none_target_sparsity
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
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
---

# YOLOv3-SPP Pruned Short

This recipe creates a sparse [YOLOv3-SPP](https://arxiv.org/abs/1804.02767) model in a shortened schedule as compared to the original pruned recipe.
It will train faster, but will recover slightly worse.
Use the following [SparseML integration with ultralytics/yolov3](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov3) to run.

When running, adjust hyperparameters based on training environment and dataset.

## Weights and Biases

- [YOLOv3-SPP LeakyReLU on VOC](https://wandb.ai/neuralmagic/yolov3-spp-lrelu-voc/runs/jktw650n)

## Training

To set up the training environment, follow the instructions on the [integration README](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/README.md).
Using the given training script from the `yolov3` directory the following command can be used to launch this recipe.  
The contents of the `hyp.pruned.yaml` hyperparameters file is given below.
Adjust the script command for your GPU device setup. 
Ultralytics supports both DataParallel and DDP.

*script command:*

```
python train.py \
    --recipe ../recipes/yolov3-spp.pruned.short.md \
    --weights PRETRAINED_WEIGHTS \
    --data voc.yaml \
    --hyp ../data/hyp.pruned.yaml \
    --name yolov3-spp-lrelu-pruned
```

hyp.pruned.yaml:
```yaml
lr0: 0.005
lrf: 0.1
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