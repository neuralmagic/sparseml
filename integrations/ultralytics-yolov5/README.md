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

This directory demonstrates how to use SparseML's `ultralytics/yolov5` integration. 

By integrating the robust training flows in the `yolov5` repository with the SparseML code base, we enable model sparsification techniques on the popular YOLOv5 architeture,
creating smaller and faster deployable versions.
The techniques include, but are not limited to:
- Pruning
- Quantization
- Sparse Transfer Learning

Once trained, SparseML enables you to export models to the ONNX format - such that they can be deployed with DeepSparse.

This integration enables spinning up one of the following end-to-end functionalities:
- **Sparse Transfer Learning** - fine-tune a pre-sparsified checkpoint on your own, private dataset.
- **Sparsification from Sractch** - easily sparsify any of the YOLOV5 and YOLOV5-P6 models, from YOLOv5n to YOLOv5x models. 

## Installation

Install with `pip`:

```pip install sparseml[torchvision]```

Note: YOLOv5 will not immediately install with this command. Instead, a sparsification-compatible version of YOLOv5 will install on the first invocation of the YOLOv5 code in SparseML.

## Tutorials

- [Sparse Transfer Learning With YOLOv5](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/tutorials/yolov5_sparse_transfer_learning.md)

- [Sparsifying YOLOv5 Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/tutorials/sparsifying_yolov5_using_recipes.md)

## Quick Tour

### Recipes

Recipes are YAML files that encode the instructions and hyperparameters for sparsifying a model using modifiers to the training process. The modifiers can range from pruning and quantization to learning rate and weight decay.

The SparseML system parses the declarative recipes as inputs and applies their
instuctions during the training process.

### SparseZoo

Neural Magic has pre-sparsified each version of YOLOv5. These models can be deployed directly or can be fine-tuned onto custom dataset via Sparse Transfer Learning. This
makes it easy to create a sparse YOLOv5 model trained on your dataset.

Check out the model cards in the [SparseZoo](https://sparsezoo.neuralmagic.com/?repo=ultralytics&page=1).

### SparseML CLI

The SparseML installation provides a CLI for running YOLOv5 scripts with SparseML capability. The full set of commands is included below

```bash
sparseml.yolov5.train
sparseml.yolov5.validation
sparseml.yolov5.export_onnx
```

Appending the `--help` argument displays a full list of options for the command:
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

## Getting Started

### Sparse Transfer Learning

Let's walk through a quick example fine-tuning a pre-sparsified YOLOv5 model onto
a new dataset.

In SparseZoo, there is a [75% pruned-quantized YOLOv5s](https://sparsezoo.neuralmagic.com/models/cv%2Fdetection%2Fyolov5-s%2Fpytorch%2Fultralytics%2Fcoco%2Fpruned75_quant-none) model, which is identified by the following SparseZoo stub:

```
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none
```

Run the following to fine-tune the model onto the the `VOC` dataset.
```bash
sparseml.yolov5.train \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn \
  --recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned75_quant-none?recipe_type=transfer_learn \
  --data VOC.yaml \
  --cfg yolov5s.yaml \
  --patience 0 \
  --hyp hyps/hyp.finetune.yaml
```

At the end, you will have a 75% pruned and quantized YOLOv5s trained on VOC!

### Export to ONNX

The SparseML installation provides a `sparseml.yolov5.export_onnx` command that you can use to create an ONNX . The export process is modified such that the quantized and pruned models are corrected and folded properly. Be sure the `--weights` argument points to your trained model. 
```bash
sparseml.yolov5.export_onnx \
    --weights path/to/weights.pt \
    --dynamic 
```

### DeepSparse Deployment

Once exported to ONNX, you can deploy your models with DeepSparse.

Checkout the DeepSparse repo for examples.