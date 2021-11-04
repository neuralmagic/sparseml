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
num_epochs: &num_epochs 60

# Pruning Hyperparams
init_sparsity: &init_sparsity 0.2
pruning_start_epoch: &pruning_start_epoch 0
pruning_end_epoch: &pruning_end_epoch 30
update_frequency: &pruning_update_frequency 0.2
mask_type: &mask_type [1, 4]
prune_none_target_sparsity: &prune_none_target_sparsity 0.6
prune_low_target_sparsity: &prune_low_target_sparsity 0.7
prune_mid_target_sparsity: &prune_mid_target_sparsity 0.8
prune_high_target_sparsity: &prune_high_target_sparsity 0.85

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: *num_epochs

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

    init_sparsity: *init_sparsity
    final_sparsity: *prune_none_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
    mask_type: *mask_type

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
    init_sparsity: *init_sparsity
    final_sparsity: *prune_low_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
    mask_type: *mask_type

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
      - num_epochs: &num_epochs 4

quantization_start_epoch: &quantization_start_epoch 0
quantization_init_lr: &quantization_init_lr 0.00001

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: *num_epochs
  - !SetLearningRateModifier
    start_epoch: *quantization_start_epoch
    learning_rate: *quantization_init_lr
  - !SetWeightDecayModifier
    start_epoch: *quantization_start_epoch
    weight_decay: 0.0


pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__

quantization_modifiers:
  - !QuantizationModifier
    start_epoch: *quantization_start_epoch
prediction_layers.0.mask_layer.weight
      - semantic_seg_conv.weight

    init_sparsity: *init_sparsity
    final_sparsity: *prune_mid_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
    mask_type: *mask_type

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
    init_sparsity: *init_sparsity
    final_sparsity: *prune_high_target_sparsity
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
    mask_type: *mask_type
      
---

# YOLACT Pruned

This recipe creates a sparse, [YOLACT](https://github.com/dbolya/yolact) model that achieves [TODO:fill recovery] 
recovery of its baseline accuracy on the COCO dataset 
( 50.16, 46.57 mAP@0.5 baseline vs 50.18, 46.80 mAP@0.5 for this recipe for bounding box, mask).
Training was done using 4 GPUs at half precision with a total batch size of 64 using the [SparseML integration with dbolya/yolact](../).
When running, adjust hyperparameters based on training environment and dataset.

## Training

To set up the training environment, follow the instructions on the [integration README](../README.md).
Using the given training script from the `yolact` directory the following command can be used to launch this recipe. 
Adjust the script command for your GPU device setup. 
YOLACT supports DDP. Currently this repo only supports YOLACT models with a darknet53 backbone.

*script command:*

```
python train.py \
--config=yolact_darknet53_config \
--recipe=./recipes/yolact.pruned.md \
--resume=PRETRAINED_WEIGHTS \
--cuda=True \
--start_iter=0 \
--batch_size=8
```
