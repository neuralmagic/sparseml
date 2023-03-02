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

# SparseML YOLOv5 Integration

By integrating with robust training flows in the YOLOv5 repository, SparseML enables you to train inference-optimized sparse versions of YOLOv5 models on your dataset.

There are two pathways:
- **Sparse Transfer Learning** - fine-tune a pre-sparsified YOLOv5 checkpoint on your own dataset **[RECOMMENDED]**
- **Sparsification from Sractch** - apply pruning and quantization to sparsify any of the YOLOv5 and YOLOv5-P6 models from scratch.

Once trained, SparseML enables you to export models to the ONNX format, such that they can be deployed with DeepSparse for GPU-class performance on the CPU.

## Installation

Install with `pip`:

```bash
pip install sparseml[torchvision]
```

**Note**: YOLOv5 will not immediately install with this command. Instead, a sparsification-compatible version of YOLOv5 will install on the first invocation of the YOLOv5 code in SparseML.

## Tutorials

- [Sparse Transfer Learning with the CLI](tutorials/sparse-transfer-learning.md) **[HIGHLY RECOMMENDED]**
- [Sparsifying From Scratch with the CLI](tutorials/sparsification-from-scratch.md)

## Quick Tour

### SparseZoo

Neural Magic has pre-sparsified each version of YOLOv5. These models can be deployed directly or can be fine-tuned onto custom dataset via sparse transfer learning. This
makes it easy to create a sparse YOLOv5 model trained on your dataset.

Check out the model cards in the [SparseZoo](https://sparsezoo.neuralmagic.com/?repo=ultralytics&page=1).

### Recipes

SparseML Recipes are YAML files that encode the instructions for sparsifying a model or sparse transfer learning. The SparseML YOLOv5 training script accepts the recipes as inputs, parses the instructions, and applies the specified algorithms and hyperparameters during the training process.

### SparseML CLI

SparseML's CLI enables you to kick-off sparsification workflows with various utilities like creating training pipelines, dataset loading, checkpoint saving, metric reporting, and logging handled for you. Appending the `--help` argument will provide a full list of options for training in SparseML:

```bash
sparseml.yolov5.train --help
```

output:
```
usage: sparseml.yolov5.train [-h] [--weights WEIGHTS] [--cfg CFG] [--data DATA] [--hyp HYP] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--imgsz IMGSZ] [--rect]
                             [--resume [RESUME]] [--nosave] [--noval] [--noautoanchor] [--evolve [EVOLVE]] [--bucket BUCKET] [--cache [CACHE]] [--image-weights]
                             [--device DEVICE] [--multi-scale] [--single-cls] [--optimizer {SGD,Adam,AdamW}] [--sync-bn] [--workers WORKERS] [--project PROJECT]
                             [--name NAME] [--exist-ok] [--quad] [--cos-lr] [--label-smoothing LABEL_SMOOTHING] [--patience PATIENCE] [--freeze FREEZE [FREEZE ...]]
                             [--save-period SAVE_PERIOD] [--local_rank LOCAL_RANK] [--entity ENTITY] [--upload_dataset [UPLOAD_DATASET]]
                             [--bbox_interval BBOX_INTERVAL] [--artifact_alias ARTIFACT_ALIAS] [--recipe RECIPE] [--disable-ema] [--max-train-steps MAX_TRAIN_STEPS]
                             [--max-eval-steps MAX_EVAL_STEPS] [--one-shot] [--num-export-samples NUM_EXPORT_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     initial weights path
  --cfg CFG             model.yaml path
  --data DATA           dataset.yaml path
  --hyp HYP             hyperparameters path
  --epochs EPOCHS
  --batch-size BATCH_SIZE
                        total batch size for all GPUs, -1 for autobatch
...
```

## Quick Start: Sparse Transfer Learning

### Overview 
Sparse Transfer is quite similiar to the typical transfer learing process used to train YOLOv5 models, where we fine-tune a pretrained checkpoint onto a smaller downstream dataset. With Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

In this example, we will fine-tune a [75% pruned-quantized version of YOLOv5s](https://sparsezoo.neuralmagic.com/models/cv%2Fdetection%2Fyolov5-s%2Fpytorch%2Fultralytics%2Fcoco%2Fpruned75_quant-none) onto VOC. 

### Kick off Training

We can start Sparse Transfer Learning by passing a starting checkpoint and recipe to the training script. For Sparse Transfer, we will use a recipe that instructs SparseML to maintain sparsity during training and to quantize the model. The starting checkpoint and transfer recipe are specified by the following SparseZoo stub:

```bash
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn
```

<details>
   <summary>Click to see the recipe</summary>
   
SparseML parses the `Modifers` in the recipe and updates the training loop with logic encoded therein.
   
The key `Modifiers` for sparse transfer learning are the following:
- `ConstantPruningModifier` instructs SparseML to maintain the sparsity structure of the network during the fine-tuning process
- `QuantizationModifier` instructs SparseML to apply quantization aware training to quantize the weights over the final epochs
   
```yaml
version: 1.1.0

# General variables
num_epochs: 55
quantization_epochs: 5
quantization_lr: 1.e-5
final_lr: 1.e-9

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)

  - !LearningRateFunctionModifier
    start_epoch: eval(num_epochs - quantization_epochs)
    end_epoch: eval(num_epochs)
    lr_func: cosine
    init_lr: eval(quantization_lr)
    final_lr: eval(final_lr)

pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__

quantization_modifiers:
  - !QuantizationModifier
    start_epoch: eval(num_epochs - quantization_epochs)
    submodules:
      - model
    custom_quantizable_module_types: ['SiLU']
    exclude_module_types: ['SiLU']
    quantize_conv_activations: False
    disable_quantization_observer_epoch: eval(num_epochs - quantization_epochs + 2)
    freeze_bn_stats_epoch: eval(num_epochs - quantization_epochs + 1)
```
   
</details>

Run the following to transfer learn from the 75% pruned-quantized YOLOv5s onto VOC.
```bash
sparseml.yolov5.train \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn \
  --recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn \
  --data VOC.yaml \
  --hyp hyps/hyp.finetune.yaml --cfg yolov5s.yaml --patience 0
```

The script uses the SparseZoo stubs to identify and download the starting checkpoint and YAML-based recipe file from the SparseZoo. SparseML parses the transfer learning recipe and adjusts the trainign process to maintain sparsity during the fine-tuning process.

The resulting model is 75% pruned and quantized and is trained on VOC!

To transfer learn this sparsified model to other datasets you may have to adjust certain hyperparameters in this recipe and/or training script such as the optimizer type, the number of epochs, and the learning rates.

### Export to ONNX

The SparseML installation provides a `sparseml.yolov5.export_onnx` command that you can use to export the model to ONNX. Be sure the `--weights` argument points to your trained model.

```bash
sparseml.yolov5.export_onnx \
    --weights path/to/weights.pt \
    --dynamic 
```

### DeepSparse Deployment

Once exported to ONNX, you can deploy your models with DeepSparse. Checkout the DeepSparse repo for examples.
