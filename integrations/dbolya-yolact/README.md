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

# SparseML YOLACT Integration

This directory combines the [SparseML](../../) recipe-driven approach with the 
[dbolya/yolact](https://github.com/dbolya/yolact) repository.
By integrating the training flows in the `yolact` repository with the SparseML 
code base,
we enable model sparsification techniques on the popular 
[YOLACT architecture](https://arxiv.org/abs/1804.02767)
creating smaller and faster deployable versions.
The techniques include, but are not limited to:

- Pruning
- Quantization
- Pruning and Quantization
- Sparse Transfer Learning

## Installation

We recommend using a [virtual environment](https://docs.python.org/3/library/venv.html) to keep your project dependencies isolated.
A virtual environment can be created using the following commands:

```bash
python3 -m venv venv # create a venv virtual environment
source venv/bin/activate # activate venv
pip install --upgrade pip # upgrade pip
```
To begin, run the following command in the root directory of this integration 
(`cd integrations/dbolya-yolact`):
```bash
bash setup_integration.sh
```

The `setup_integration.sh` file will clone the yolact-sparseml integration repository. After the repo has successfully cloned, all dependencies from the `yolact/requirements.txt` file will install in your current environment.

Note: This integration requires `python>=3.7,<3.10`


## Quick Tour

### Recipes
Recipes encode the instructions and hyperparameters for sparsifying a model using modifiers to the training process.
The modifiers can range from pruning and quantization to learning rate and weight decay.
When appropriately combined, it becomes possible to create highly sparse and accurate models.

This integration adds a `--recipe` argument to the 
[`train.py` script](https://github.com/neuralmagic/yolact/blob/master/train.py).
The argument loads an appropriate recipe while preserving the rest of the training pipeline.
Popular recipes used with this argument are found in the [`recipes` folder](./recipes).
Otherwise, all other arguments and functionality remain the same as the original repository.

### SparseZoo

| Sparsification Type | Description                                                                       | Zoo Stub                                                                     | COCO mAP@all | Size on Disk | DeepSparse Performance** |
|---------------------|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------|--------------|--------------|--------------------------|
| Baseline            | The baseline, pretrained model on the COCO dataset.                               | zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none           | 0.288        | 170 MB       | -- img/sec               |
| Pruned              | A highly sparse, FP32 model that recovers close to the baseline model.            | zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned90-none       | 0.286        | 30.1 MB      | -- img/sec               |
| Pruned Quantized    | A highly sparse, INT8 model that recovers reasonably close to the baseline model. | zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none | 0.282        | 9.7 MB       | -- img/sec               |

These models can also be viewed on the [SparseZoo Website](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=segmentation&page=1).

### Structure

The following table lays out the root-level files and folders along with a description for each.

| Folder/File Name                                | Description                                                                                                           |
|-------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| [recipes](./recipes)                            | Typical recipes for sparsifying YOLACT models along with any downloaded recipes from the SparseZoo.                   |
| [yolact](./yolact)                              | Integration repository folder used to train and sparsify YOLACT models (`setup_integration.sh` must run first).       |
| [README.md](./README.md)                        | Readme file.                                                                                                          |
| [setup_integration.sh](./setup_integrations.sh) | Setup file for the integration run from the command line.                                                             |
| [tutorials](./tutorials)                        | Easy to follow sparsification tutorials for YOLACT  models.                                                            |

### Exporting for Inference

After sparsifying a model, the 
[`export.py` script](https://github.com/neuralmagic/yolact/blob/master/export.py)
converts the model into deployment formats such as [ONNX](https://onnx.ai/).
The export process is modified such that the quantized and pruned models are 
corrected and folded properly.

For example, the following command can be run from within the Neural Magic's 
`yolact` repository folder to export a trained/sparsified model's checkpoint:

```bash
python export.py --checkpoint ./quantized-yolact/model.pth \
    --name quantized-yolact.onnx
```


### DeepSparse

The [DeepSparse Engine](https://github.com/neuralmagic/deepsparse) accepts ONNX 
formats and is engineered to significantly speed up inference on CPUs for 
the sparsified models from this integration. [Example](https://github.com/neuralmagic/deepsparse/tree/main/examples/dbolya-yolact) scripts can be found in the DeepSparse repository.

## Citation
```bibtex
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```
