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

# SparseML Ultralytics YOLOv3 Integration

This directory combines the SparseML recipe-driven approach with the 
[ultralytics/yolov3](https://github.com/ultralytics/yolov3) repository.
By integrating the robust training flows in the `yolov3` repository with the SparseML code base,
we enable model sparsification techniques on the popular [YOLOv3 architecture](https://arxiv.org/abs/1804.02767)
creating smaller and faster deployable versions.
The techniques include, but are not limited to:
- Pruning
- Quantization
- Pruning + Quantization
- Sparse Transfer Learning

## Highlights

- Blog: [YOLOv3 on CPUs: Sparsifying to Achieve GPU-Level Performance](https://neuralmagic.com/blog/benchmark-yolov3-on-cpus-with-deepsparse/)
- Example: [DeepSparse YOLOv3 Inference Example](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo)
- Video: [DeepSparse YOLOv3 Pruned Quantized Performance](https://youtu.be/o5qIYs47MPw)

## Tutorials

- [Sparsifying YOLOv3 Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/tutorials/sparsifying_yolov3_using_recipes.md)
- [Sparse Transfer Learning With YOLOv3](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/tutorials/yolov3_sparse_transfer_learning.md)

## Installation

To begin, run the following command in the root directory of this integration (`cd integrations/ultralytics-yolov3`):
```bash
bash setup_integration.sh
```

The setup_integration.sh file will clone the yolov3 repository with the SparseML integration as a subfolder.
After the repo has successfully cloned,  all dependencies from the `yolov3/requirements.txt` file will install in your current environment.

## Quick Tour

### Recipes

Recipes encode the instructions and hyperparameters for sparsifying a model using modifiers to the training process.
The modifiers can range from pruning and quantization to learning rate and weight decay.
When appropriately combined, it becomes possible to create highly sparse and accurate models.

This integration adds a `--recipe` argument to the [`train.py` script](https://github.com/neuralmagic/yolov3/blob/master/train.py).
The argument loads an appropriate recipe while preserving the rest of the training pipeline.
Popular recipes used with this argument are found in the [`recipes` folder](./recipes).
Otherwise, all other arguments and functionality remain the same as the original repository.

For example, pruning and quantizing a model on the VOC dataset can be done by running the following command from within the yolov3 repository folder:
```bash
python train.py --batch 16 --img 512 --weights PRETRAINED_WEIGHTS \
    --data voc.yaml --hyp ../data/hyp.pruned_quantized.yaml \
    --recipe ../recipes/yolov3-spp.pruned_quantized.short.md
```

### SparseZoo

Pre-sparsified models and recipes can be downloaded through the [SparseZoo](https://github.com/neuralmagic/sparsezoo).
The following popular stubs can be used to download these models and recipes through the SparseZoo API directly.

| Model                       | Dataset | mAP@50 | Description                                                    | SparseZoo Stub                                                                   |
|-----------------------------|---------|--------|----------------------------------------------------------------|----------------------------------------------------------------------------------|
| YOLOv3-SPP Pruned Quantized | COCO    | 60.4   | 83% sparse and INT8 quantized model with LeakyReLU activations | zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94 |
| YOLOv3-SPP Pruned           | COCO    | 62.4   | 88% sparse model with LeakyReLU activations                    | zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned-aggressive_97       |
| YOLOv3-SPP                  | COCO    | 64.2   | Baseline model with LeakyReLU activations                      | zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/base-none                  |

Complete lists are available online for all [models](https://sparsezoo.neuralmagic.com/tables/models/cv/detection?repo=ultralytics) and 
[recipes](https://sparsezoo.neuralmagic.com/tables/recipes/cv/detection?repo=ultralytics) compatible with this integration as well.

Sample code for retrieving a model from the SparseZoo:
```python
from sparsezoo import Model

model = Model("zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94")
print(model)
```

Sample code for retrieving a recipe from the SparseZoo:
```python
from sparsezoo import Model

model = Model("zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94")
recipe = model.recipes.default
print(recipe)
```

### Structure

The following table lays out the root-level files and folders along with a description for each.

| Folder/File Name     | Description                                                                                                           |
|----------------------|-----------------------------------------------------------------------------------------------------------------------|
| data                 | The hyperparameters to use and files to use for training on data.                                                     |
| models               | Model architecture definitions along with any downloaded checkpoints from the SparseZoo.                              |
| recipes              | Typical recipes for sparsifying YOLOv3 models along with any downloaded recipes from the SparseZoo.                   |
| tutorials            | Tutorial walkthroughs for how to sparsify YOLOv3 models using recipes.                                                |
| yolov3               | Integration repository folder used to train and sparsify YOLOv3 models (setup_integration.sh must run first).         |
| README.md            | Readme file.                                                                                                          |
| setup_integration.sh | Setup file for the integration run from the command line.                                                             |

### Exporting for Inference

After sparsifying a model, the [`export.py` script](https://github.com/neuralmagic/yolov3/blob/master/models/export.py) 
converts the model into deployment formats such as [ONNX](https://onnx.ai/).
The export process is modified such that the quantized and pruned models are corrected and folded properly.

For example, the following command can be run from within the yolov3 repository folder to export a trained/sparsified model's checkpoint:
```bash
python models/export.py --weights PATH/TO/weights.pt --img-size 512 512
```

The DeepSparse Engine accepts ONNX formats and is engineered to significantly speed up inference on CPUs for the sparsified models from this integration.
Examples for loading, benchmarking, and deploying can be found in the [DeepSparse repository here](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo).
