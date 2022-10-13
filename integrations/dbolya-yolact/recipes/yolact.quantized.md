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
num_epochs: 2
quantization_start_target: 0
quantization_end_target: 1
quantization_observer_end_target: 0.5
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: eval(num_epochs)

  - !SetWeightDecayModifier
    start_epoch: eval(quantization_start_target * num_epochs)
    weight_decay: 0.0

  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: eval(quantization_end_target * num_epochs)
    lr_func: linear
    init_lr: 0.00001
    final_lr: 0.000001

pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__

quantization_modifiers:
  - !QuantizationModifier
    start_epoch: eval(quantization_start_target * num_epochs)
    disable_quantization_observer_epoch: eval(quantization_observer_end_target * num_epochs)
    freeze_bn_stats_epoch: eval(quantization_observer_end_target * num_epochs)
---

# YOLACT Quantized

This recipe quantizes a [YOLACT](https://github.com/dbolya/yolact) model.
Training was done using 4 GPUs with a total batch size of 64 using the 
[SparseML integration with dbolya/yolact](../).
When running, adjust hyper-parameters based on the training environment and dataset.

## Training

To set up the training environment, follow the instructions on the [integration README](../README.md).
The following command can be used to launch this recipe. 
Adjust the script command for your GPU device setup.
YOLACT supports DataParallel. Currently, this repo only supports YOLACT models with a Darknet-53 backbone.

*script command:*

```
sparseml.yolact.train \
--recipe=../recipes/yolact.quantized.md \
--resume=zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none \
--train_info ./data/coco/annotations/instances_train2017.json \
--validation_info ./data/coco/annotations/instances_val2017.json \
--train_images ./data/coco/images \
--validation_images ./data/coco/images
```