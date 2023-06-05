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

This directory explains how to use SparseML's `ultralytics/yolov5` integration to train inference-optimized sparse YOLOv5 models on your dataset.

There are two main workflows enabled by SparseML:
- **Sparse Transfer Learning** - fine-tune a pre-sparsified YOLOv5 checkpoint on your own dataset **[RECOMMENDED]**
- **Sparsification from Scratch** - apply pruning and quantization to sparsify any YOLOv5 model from scratch

Once trained, SparseML enables you to export models to the ONNX format, such that they can be deployed with DeepSparse.

## Installation

Install with `pip`:

```bash
pip install sparseml[yolov5]
```

## Tutorials

- [Sparse Transfer Learning with the CLI](tutorials/sparse-transfer-learning.md) **[HIGHLY RECOMMENDED]**
- [Sparsifying From Scratch with the CLI](tutorials/sparsify-from-scratch.md)

## Quick Tour

### SparseZoo

SparseZoo is an open-source repository of pre-sparsified models, including each version of of YOLOv5. With SparseML, you can fine-tune these pre-sparsified checkpoints onto custom datasets (while maintaining sparsity) via sparse transfer learning. This makes training inference-optimized sparse models almost identical to your typical YOLOv5 training workflow.

[Check out the available models](https://sparsezoo.neuralmagic.com/?repo=ultralytics&page=1)

### Recipes

Recipes are YAML files that encode the instructions for sparsifying a model or sparse transfer learning. SparseML accepts the recipes as inputs, parses the instructions, and applies the specified algorithms and hyperparameters during the training process.

In this way, recipes are the declarative interface for specifying which sparsity-related algorithms to apply, allowing you to apply sparsity related algorithms from the familiar YOLOv5 training script.

### SparseML CLI

SparseML's CLI is built on top of YOLOv5's [`train.py`](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) script. This enables you to kick-off sparse training workflows with all of the friendly utilities from the friendly Ultralytics repo like dataset loading and preprocessing, checkpoint saving, metric reporting, and logging handled for you. Appending the `--help` argument will provide a full list of options for training in SparseML:

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

SparseML inherits most arguments from the Ultralytics repository. [Check out the YOLOv5 documentation for usage](https://github.com/ultralytics/yolov5).

## Quick Start: Sparse Transfer Learning

### Sparse Transfer Learning Overview 

Sparse Transfer is very similiar to the typical transfer learing process used to train YOLOv5 models, where we fine-tune a checkpoint pretrained on COCO onto a smaller downstream dataset. With Sparse Transfer Learning, however, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

Here, we will fine-tune a [75% pruned-quantized version of YOLOv5s](https://sparsezoo.neuralmagic.com/models/yolov5-s-coco-pruned75_quantized?comparison=yolov5-s-coco-base&tab=0) onto VOC. 

### Kick off Training

We will use SparseML's `sparseml.yolov5.train` training script.

To run sparse transfer learning, we first need to create/select a sparsification recipe. For sparse transfer, we need a recipe that instructs SparseML to maintain sparsity during training and to quantize the model over the final epochs.

For the VOC dataset, there is a [transfer learning recipe available in SparseZoo](https://sparsezoo.neuralmagic.com/models/yolov5-s-coco-pruned75_quantized?comparison=yolov5-s-coco-base&tab=0), found under the recipes tab and identified by the following SparseZoo stub:
```bash
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn
```

Here is what the recipe looks like:
   
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
   
The key `Modifiers` for sparse transfer learning are the following:
- `ConstantPruningModifier` instructs SparseML to maintain the sparsity structure of the network during the fine-tuning process
- `QuantizationModifier` instructs SparseML to apply quantization aware training to quantize the weights over the final epochs

SparseML parses the `Modifers` in the recipe and updates the training loop with logic encoded therein.

Run the following to transfer learn from the 75% pruned-quantized YOLOv5s onto VOC:
```bash
sparseml.yolov5.train \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn \
  --recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn \
  --data VOC.yaml \
  --hyp hyps/hyp.finetune.yaml --cfg yolov5s.yaml --patience 0
```

The script uses the SparseZoo stubs to identify and download the starting checkpoint and YAML-based recipe file from the SparseZoo. SparseML parses the transfer learning recipe and adjusts the training process to maintain sparsity during the fine-tuning process. It then kicks off the training process.

The resulting model is 75% pruned and quantized and is trained on VOC!

### Export to ONNX

Run the `sparseml.yolov5.export_onnx` command to export the model to ONNX. Be sure the `--weights` argument points to your trained model.

```bash
sparseml.yolov5.export_onnx \
  --weights yolov5_runs/train/exp/weights/last.pt \
  --dynamic 
```

### DeepSparse Deployment

Once exported to ONNX, you can deploy your models with DeepSparse. Checkout the [DeepSparse Repository](https://github.com/neuralmagic/deepsparse) for more details.

## Next Steps

Check out the tutorials for more details on additional functionality like training with other YOLOv5 versions and using custom datasets:

- [Sparse Transfer Learning with the CLI](tutorials/sparse-transfer-learning.md) **[HIGHLY RECOMMENDED]**
- [Sparsifying From Scratch with the CLI](tutorials/sparsify-from-scratch.md)
