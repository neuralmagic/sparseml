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

# SparseML Ultralytics YOLOv5 Integration

This directory combines the SparseML recipe-driven approach with the 
[ultralytics/yolov5](https://github.com/ultralytics/yolov5) repository.
By integrating the robust training flows in the `yolov5` repository with the SparseML code base,
we enable model sparsification techniques on the popular [YOLOv5 architecture](https://github.com/ultralytics/yolov5/issues/280)
creating smaller and faster deployable versions.
The techniques include, but are not limited to:
- Pruning
- Quantization
- Pruning + Quantization
- Sparse Transfer Learning

## Highlights

- Example: [DeepSparse YOLOv5 Inference Example](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo)

## Tutorials

- [Sparsifying YOLOv5 Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/tutorials/sparsifying_yolov5_using_recipes.md)
- [Sparse Transfer Learning With YOLOv5](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/tutorials/yolov5_sparse_transfer_learning.md)

## Installation

To begin, run the following command in the root directory of this integration (`cd integrations/ultralytics-yolov5`):
```bash
bash setup_integration.sh
```

The setup_integration.sh file will clone the yolov5 repository with the SparseML integration as a subfolder.
After the repo has successfully cloned,  all dependencies from the `yolov5/requirements.txt` file will install in your current environment.
Note, the `yolov5` repository requires Python 3.7 or greater.

## Quick Tour

### Recipes

Recipes encode the instructions and hyperparameters for sparsifying a model using modifiers to the training process.
The modifiers can range from pruning and quantization to learning rate and weight decay.
When appropriately combined, it becomes possible to create highly sparse and accurate models.

This integration adds a `--recipe` argument to the [`train.py` script](https://github.com/neuralmagic/yolov5/blob/master/train.py).
The argument loads an appropriate recipe while preserving the rest of the training pipeline.
Popular recipes used with this argument are found in the [`recipes` folder](./recipes).
Otherwise, all other arguments and functionality remain the same as the original repository.

### SparseZoo

SparseZoo models are coming soon!

### Structure

The following table lays out the root-level files and folders along with a description for each.

| Folder/File Name     | Description                                                                                                           |
|----------------------|-----------------------------------------------------------------------------------------------------------------------|
| data                 | The hyperparameters to use and files to use for training on data.                                                     |
| models               | Model architecture definitions along with any downloaded checkpoints from the SparseZoo.                              |
| recipes              | Typical recipes for sparsifying YOLOv5 models along with any downloaded recipes from the SparseZoo.                   |
| tutorials            | Tutorial walkthroughs for how to sparsify YOLOv5 models using recipes.                                                |
| yolov5               | Integration repository folder used to train and sparsify YOLOv5 models (setup_integration.sh must run first).         |
| README.md            | Readme file.                                                                                                          |
| setup_integration.sh | Setup file for the integration run from the command line.                                                             |

### Exporting for Inference

After sparsifying a model, the [`export.py` script](https://github.com/neuralmagic/yolov5/blob/master/models/export.py) 
converts the model into deployment formats such as [ONNX](https://onnx.ai/).
The export process is modified such that the quantized and pruned models are corrected and folded properly.

For example, the following command can be run from within the yolov5 repository folder to export a trained/sparsified model's checkpoint:
```bash
python models/export.py --weights PATH/TO/weights.pt --dynamic
```

The DeepSparse Engine accepts ONNX formats and is engineered to significantly speed up inference on CPUs for the sparsified models from this integration.
Examples for loading, benchmarking, and deploying can be found in the [DeepSparse repository here](https://github.com/neuralmagic/deepsparse/tree/main/examples/ultralytics-yolo).
