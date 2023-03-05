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
# General Hyperparams
num_epochs: 62
init_lr: 0.001
num_quantization_epochs: 2
num_pruning_epochs: eval(num_epochs - num_quantization_epochs)
quantization_observer_end_target: 0.5

# Pruning Hyperparams
init_sparsity: 0.2
pruning_start_target: 0
pruning_end_target: 0.5
pruning_update_frequency: 0.2
mask_type: [1, 4]
prune_none_target_sparsity: 0.6
prune_low_target_sparsity: 0.7
prune_mid_target_sparsity: 0.8
prune_high_target_sparsity: 0.85

# quantization hyperparams

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)

  # pruning
  - !SetWeightDecayModifier
    start_epoch: eval(pruning_start_target * num_pruning_epochs)
    weight_decay: 0.0005

  - !LearningRateFunctionModifier
    start_epoch: eval(pruning_start_target * num_pruning_epochs)
    end_epoch: eval(pruning_end_target * num_pruning_epochs)
    lr_func: linear
    init_lr: eval(init_lr)
    final_lr: eval(0.01 * init_lr)
  # quantization
  - !SetWeightDecayModifier
    start_epoch: eval(num_epochs - num_quantization_epochs)
    weight_decay: 0.0

  - !LearningRateFunctionModifier
    start_epoch: eval(num_epochs - num_quantization_epochs)
    end_epoch: eval(num_epochs)
    lr_func: linear
    init_lr: eval(0.01 * init_lr)
    final_lr: eval(0.001 * init_lr)

pruning_modifiers:
  - !GMPruningModifier
    params:
    - backbone.layers.0.1.conv1.0.weight
    - backbone.layers.0.1.conv2.0.weight
    - backbone.layers.1.0.0.weight
    - backbone.layers.1.1.conv1.0.weight
    - backbone.layers.1.1.conv2.0.weight
    - backbone.layers.1.2.conv1.0.weight
    - backbone.layers.1.2.conv2.0.weight

    init_sparsity: eval(init_sparsity)
    final_sparsity: eval(prune_none_target_sparsity)
    start_epoch: eval(pruning_start_target * num_pruning_epochs)
    end_epoch: eval(pruning_end_target * num_pruning_epochs)
    update_frequency: eval(pruning_update_frequency)
    mask_type: eval(mask_type)

  - !GMPruningModifier
    params:
      - backbone.layers.2.0.0.weight
      - backbone.layers.2.1.conv1.0.weight
      - backbone.layers.2.1.conv2.0.weight
      - backbone.layers.2.2.conv1.0.weight
      - backbone.layers.2.2.conv2.0.weight
      - backbone.layers.2.3.conv1.0.weight
      - backbone.layers.2.3.conv2.0.weight
      - backbone.layers.2.4.conv1.0.weight
      - backbone.layers.2.4.conv2.0.weight
      - backbone.layers.2.5.conv1.0.weight
      - backbone.layers.2.5.conv2.0.weight
      - backbone.layers.2.6.conv1.0.weight
      - backbone.layers.2.6.conv2.0.weight
      - backbone.layers.2.7.conv1.0.weight
      - backbone.layers.2.7.conv2.0.weight
      - backbone.layers.2.8.conv1.0.weight
      - backbone.layers.2.8.conv2.0.weight
    init_sparsity: eval(init_sparsity)
    final_sparsity: eval(prune_low_target_sparsity)
    start_epoch: eval(pruning_start_target * num_pruning_epochs)
    end_epoch: eval(pruning_end_target * num_pruning_epochs)
    update_frequency: eval(pruning_update_frequency)
    mask_type: eval(mask_type)

  - !GMPruningModifier
    params:
      - proto_net.0.weight
      - proto_net.2.weight
      - proto_net.4.weight
      - proto_net.8.weight
      - proto_net.10.weight
      - fpn.lat_layers.0.weight
      - fpn.lat_layers.1.weight
      - fpn.lat_layers.2.weight
      - fpn.pred_layers.0.weight
      - fpn.pred_layers.1.weight
      - fpn.pred_layers.2.weight
      - fpn.downsample_layers.0.weight
      - fpn.downsample_layers.1.weight
      - prediction_layers.0.upfeature.0.weight
      - prediction_layers.0.bbox_layer.weight
      - prediction_layers.0.conf_layer.weight
      - prediction_layers.0.mask_layer.weight
      - semantic_seg_conv.weight

    init_sparsity: eval(init_sparsity)
    final_sparsity: eval(prune_mid_target_sparsity)
    start_epoch: eval(pruning_start_target * num_pruning_epochs)
    end_epoch: eval(pruning_end_target * num_pruning_epochs)
    update_frequency: eval(pruning_update_frequency)
    mask_type: eval(mask_type)

  - !GMPruningModifier
    params:
      - backbone.layers.3.0.0.weight
      - backbone.layers.3.1.conv1.0.weight
      - backbone.layers.3.1.conv2.0.weight
      - backbone.layers.3.2.conv1.0.weight
      - backbone.layers.3.2.conv2.0.weight
      - backbone.layers.3.3.conv1.0.weight
      - backbone.layers.3.3.conv2.0.weight
      - backbone.layers.3.4.conv1.0.weight
      - backbone.layers.3.4.conv2.0.weight
      - backbone.layers.3.5.conv1.0.weight
      - backbone.layers.3.5.conv2.0.weight
      - backbone.layers.3.6.conv1.0.weight
      - backbone.layers.3.6.conv2.0.weight
      - backbone.layers.3.7.conv1.0.weight
      - backbone.layers.3.7.conv2.0.weight
      - backbone.layers.3.8.conv1.0.weight
      - backbone.layers.3.8.conv2.0.weight
      - backbone.layers.4.0.0.weight
      - backbone.layers.4.1.conv1.0.weight
      - backbone.layers.4.1.conv2.0.weight
      - backbone.layers.4.2.conv1.0.weight
      - backbone.layers.4.2.conv2.0.weight
      - backbone.layers.4.3.conv1.0.weight
      - backbone.layers.4.3.conv2.0.weight
      - backbone.layers.4.4.conv1.0.weight
      - backbone.layers.4.4.conv2.0.weight
    init_sparsity: eval(init_sparsity)
    final_sparsity: eval(prune_high_target_sparsity)
    start_epoch: eval(pruning_start_target * num_pruning_epochs)
    end_epoch: eval(pruning_end_target * num_pruning_epochs)
    update_frequency: eval(pruning_update_frequency)
    mask_type: eval(mask_type)

quantization_modifiers:
  - !QuantizationModifier
    start_epoch: eval(num_epochs - num_quantization_epochs)
    disable_quantization_observer_epoch: eval(num_epochs - (quantization_observer_end_target * num_quantization_epochs))
    freeze_bn_stats_epoch: eval(num_epochs - (quantization_observer_end_target * num_quantization_epochs))
---

# YOLACT Pruned Quantized

This recipe creates a sparse quantized, [YOLACT](https://github.com/dbolya/yolact) model that achieves about 
98% recovery on the COCO dataset when compared against baseline ( 30.61, 28.80 mAP@all baseline Vs 29.82, 28.26 mAP@all for this recipe for bounding box, mask).
Training was done using 4 GPUs with a total batch size of 64 using the [SparseML integration with dbolya/yolact](../).
When running, adjust hyper-parameters based on the training environment and dataset.

## Training

To set up the training environment, follow the instructions on the [integration README](../README.md).
The following command can be used to launch this recipe. 
Adjust the script command for your GPU device setup.
YOLACT supports DataParallel. Currently, this repo only supports YOLACT models with a Darknet-53 backbone.

*script command:*

```
sparseml.yolact.train \
--recipe=../recipes/yolact.pruned_quant.md \
--resume=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none \
--train_info ./data/coco/annotations/instances_train2017.json \
--validation_info ./data/coco/annotations/instances_val2017.json \
--train_images ./data/coco/images \
--validation_images ./data/coco/images
```
