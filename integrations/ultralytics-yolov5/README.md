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
This directory combines the SparseML recipe-driven approach with the 
[ultralytics/yolov5](https://github.com/ultralytics/yolov5) repository.
By integrating the robust training flows in the `yolov5` repository with the SparseML code base, we enable model sparsification techniques on the popular [YOLOv5 architecture](https://github.com/ultralytics/yolov5/issues/280)
as well as the updated [YOLOV5-P6 architecture](https://github.com/ultralytics/yolov5/releases/tag/v5.0),
creating smaller and faster deployable versions.
The techniques include, but are not limited to:
- Pruning
- Quantization
- Sparse Transfer Learning

After training, the model can be deployed with Neural Magic's DeepSparse Engine. The engine enables inference with GPU-class performance directly on your CPU.

This integration enables spinning up one of the following end-to-end functionalities:
- **Sparsification of YOLOv5 Models** - easily sparsify any of the YOLOV5 and YOLOV5-P6 models, from YOLOv5n to YOLOv5x models. 
- **Sparse Transfer Learning** - fine-tune a sparse backbone model (or use one of our [sparse pre-trained models](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=detection&page=1)) on your own, private dataset.

## Installation

```pip install sparseml[torchvision]```

Note: YOLOv5 will not immediately install with this command. Instead, a sparsification-compatible version of YOLOv5 will install on the first invocation of the YOLOv5 code in SparseML.

## Tutorials

- [Sparsifying YOLOv5 Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/tutorials/sparsifying_yolov5_using_recipes.md)
- [Sparse Transfer Learning With YOLOv5](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/tutorials/yolov5_sparse_transfer_learning.md)

## Quick Tour

### Recipes

Recipes encode the instructions and hyperparameters for sparsifying a model using modifiers to the training process.
The modifiers can range from pruning and quantization to learning rate and weight decay.
When appropriately combined, it becomes possible to create highly sparse and accurate models.

### SparseZoo

Neural Magic’s ML team creates sparsified models that allow anyone to plug in their data and leverage pre-sparsified models from the SparseZoo.
Select a YOLOv5 model from the [SparseZoo](https://sparsezoo.neuralmagic.com/?repo=ultralytics&page=1).

### SparseML CLI

The SparseML installation provides a CLI for running YOLOv5 scripts with SparseML capability. The full set of commands is included below

```bash
sparseml.yolov5.train
sparseml.yolov5.validation
sparseml.yolov5.export_onnx
sparseml.yolov5.val_onnx
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

### Sparsifying YOLOv5
In the example below, a YOLOv5s model pre-trained on COCO is pruned and quantized with recipe 
`zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94` (located in [SparseZoo](https://sparsezoo.neuralmagic.com/models/cv%2Fdetection%2Fyolov5-s%2Fpytorch%2Fultralytics%2Fcoco%2Fpruned_quant-aggressive_94)) while continuing training on COCO. You may sparsify your model while training on your own, private (downstream) dataset or while continuing training with the original (upstream) dataset.  

```bash
sparseml.yolov5.train \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none \
  --data coco.yaml \
  --hyp data/hyps/hyp.scratch.yaml \
  --recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94 
```

### Sparse Transfer Learning

Once you sparsify a model using SparseML, you can easily sparse fine-tune it on a new dataset.
While you are free to use your backbone, we encourage you to leverage one of our [sparse pre-trained models](https://sparsezoo.neuralmagic.com) to boost your productivity!

In the example below, we fetch a pruned, quantized YOLOv5s model, pre-trained on the COCO dataset. We then fine-tune the model to the VOC dataset. 
```bash
sparseml.yolov5.train \
  --data VOC.yaml \
  --cfg models_v5.0/yolov5s.yaml \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94?recipe_type=transfer \
  --hyp data/hyps/hyp.finetune.yaml \
  --recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96
```

## Once the Training is Done...

### Exporting the Sparse Model to ONNX
The DeepSparse Engine accepts ONNX formats and is engineered to significantly speed up inference on CPUs for the sparsified models from this integration.

The SparseML installation provides a `sparseml.yolov5.export_onnx` command that you can use to load the training model folder and create a new `model.onnx` file within. The export process is modified such that the quantized and pruned models are corrected and folded properly. Be sure the `--weights` argument points to your trained model. 
```bash
sparseml.yolov5.export_onnx \
    --weights path/to/weights.pt \
    --dynamic 
```

### DeepSparse Engine Benchmarking and Deployment

Once the model is exported to the ONNX format, it is ready for deployment with the DeepSparse Engine. 

To benchmark the model, run

```bash
deepsparse.benchmark path/to/model.onnx
```

To run validation in the DeepSparse Engine, execute

```bash
sparseml.yolov5.val_onnx --model_path path/to/model.onnx --data coco128.yaml
```


To learn more about the engine and deploying your model, refer to the [appropriate documentation in the DeepSparse repository](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/yolo).

## Support

For Neural Magic Support, sign up or log in to our [Deep Sparse Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue](https://github.com/neuralmagic/sparseml/issues).