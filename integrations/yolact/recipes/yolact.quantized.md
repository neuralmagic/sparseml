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
num_epochs: &num_epochs 4

# Quantization Hyperparams
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
---
# YOLACT Pruned

This recipe quantizes a [YOLACT](https://github.com/dbolya/yolact) model.
Training was done using 4 GPUs at half precision with a total batch size of 64 using the 
[SparseML integration with dbolya/yolact](../).
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
--recipe=./recipes/yolact.quantize.md \
--resume=PRETRAINED_WEIGHTS \
--cuda=True \
--start_iter=0 \
--batch_size=8
```