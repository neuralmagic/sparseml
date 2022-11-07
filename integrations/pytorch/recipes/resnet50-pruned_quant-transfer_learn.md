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
# epoch and learning rate variables
num_epochs: &num_epochs 10.0
init_lr: &init_lr 0.008
lr_step_milestones: &lr_step_milestones [5]

# quantization variables
quantization_start_epoch: &quantization_start_epoch 4.0
quantize_submodules: &quantize_submodules
  - input
  - sections

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: *num_epochs
    
  - !LearningRateModifier
    start_epoch: 0.0
    lr_class: MultiStepLR
    lr_kwargs:
      milestones: *lr_step_milestones
      gamma: 0.1
    init_lr: *init_lr

# phase 1 sparse transfer learning / recovery
sparse_transfer_learning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL__

# phase 2 apply quantization
sparse_quantized_transfer_learning_modifiers:
  - !QuantizationModifier
    start_epoch: *quantization_start_epoch
    submodules:
      - input
      - sections

  - !SetWeightDecayModifier
    start_epoch: 0.0
    weight_decay: 0.0
---

# Transfer Learning ResNet50 ImageNet Pruned-Quantized

This recipe provides a framework for performing pruned-quantized transfer learning on
SparseML's `pruned_quantized` ResNet50 model.  This recipe was written for transfer learning
onto the [Imagenette](https://github.com/fastai/imagenette) dataset using
a single GPU, Adam optimizer, and batch size 32.  Adjust the hyperparameters at the top of the file accordingly
for your dataset and training environment.

## Imagenette
On the default transfer learning dataset, Imagenette, this recipe fully recovered to 99.9% Top-1 accuracy.
To run this recipe, you may run the example notebook
`sparseml/integrations/pytorch/notebooks/sparse_quantized_transfer_learning.ipynb`
or can run the example script command below.

```
sparseml.image_classification.train \
    --recipe-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate?recipe_type=transfer_learn \
    --checkpoint-path zoo \
    --arch-key resnet50 \
    --model-kwargs '{"ignore_error_tensors": ["classifier.fc.weight", "classifier.fc.bias"]}' \
    --dataset imagenette \
    --dataset-path /PATH/TO/IMAGENETTE  \
    --train-batch-size 32 --test-batch-size 64 \
    --loader-num-workers 8 \
    --optim Adam \
    --optim-args '{}' \
    --model-tag resnet50-imagenette-pruned_quant-transfer_learned
```

## Other Datasets
To transfer learn this ResNet 50 Pruned-Quantized model to other datasets
you may have to adjust certain hyperparameters in this recipe and/or training script.
Some considerations:
* For more complex datasets, increate the number of epochs, adjusting the learning rate step accordingly
* Adding more learning rate step milestones can lead to more jumps in accuracy
* Increasing the number of epochs before starting quantization can give a higher baseline
* If accuracy is not recovering in the quantization phase, the total number of epochs should be increased as well
* Increase the learning rate when increasing batch size
* Increase the number of epochs if using SGD instead of the Adam optimizer
* Update the base learning rate based on the number of steps needed to train your dataset 
