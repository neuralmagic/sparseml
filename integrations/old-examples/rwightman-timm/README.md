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

# SparseML-rwightman/pytorch-image-models integration
This directory provides a SparseML integrated training script for the popular
[rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
repository also known as [timm](https://pypi.org/project/timm/).

Using this integration, you will be able to apply SparseML optimizations
to the powerful training flows of the pytorch-image-models repository.

Some of the tasks you can perform using this integration include, but are not limited to:
* Pruning
* Quantization
* Pruning and Quantization
* Sparse Transfer Learning

## Installation
We recommend using a [virtual environment](https://docs.python.org/3/library/venv.html) to keep your project dependencies isolated.
A virtual environment can be created using the following commands:

```bash
python3 -m venv venv # create a venv virtual environment
source venv/bin/activate # activate venv
pip install --upgrade pip # upgrade pip
```

To begin, run the following command in the root directory of this integration (`cd integrations/rwightman-timm`):
```bash
bash setup_integration.sh
```

The `setup_integration.sh` file will clone the [pytorch-image-models](https://github.com/neuralmagic/pytorch-image-models.git) repository with the SparseML integration as a subfolder.
After the repo has successfully cloned, pytorch-image-models will be installed along with any necessary dependencies.

This integration requires `python>=3.7,<3.10`

## Quick Tour
### Recipes
Recipes encode the instructions and hyperparameters for sparsifying a model using modifiers to the training process.
The modifiers can range from pruning and quantization to learning rate and weight decay.
When appropriately combined, it becomes possible to create highly sparse and accurate models.

This integration adds a `--recipe` argument to the 
[`train.py` script](https://github.com/neuralmagic/pytorch-image-models/blob/master/train.py).
The argument loads an appropriate recipe while preserving the rest of the training pipeline.
Popular recipes used with this argument are found in the [`recipes` folder](./recipes).
Otherwise, all other arguments and functionality remain the same as the original repository.

You can learn how to build or download a recipe using the
[SparseML](https://github.com/neuralmagic/sparseml)
or [SparseZoo](https://github.com/neuralmagic/sparsezoo)
documentation, or export one with [Sparsify](https://github.com/neuralmagic/sparsify).

ViT recipes are not yet available on SparseZoo, but are in progress.

Using a local recipe and checkpoint, pruning a model can be done by running the following command from within the root of this integration's folder
```bash
python train.py \
  /PATH/TO/DATASET/imagenet/ \
  --recipe ../recipes/vit_base.85.recal.config.yaml \
  --dataset imagenet \
  --batch-size 64 \
  --remode pixel --reprob 0.0 --smoothing 0.1 --mixup 0.5 --mixup 0.5 \
  --output models/optimized \
  --model vit_base_patch16_224 \
  --workers 8 
```  

Documentation on the original script can be found
[here](https://huggingface.co/docs/timm/training_script).
The latest commit hash that `train.py` is based on is included in the docstring.


The following table lays out the root-level files and folders along with a description for each.

| Folder/File Name     | Description                                                                                                           |
|----------------------|-----------------------------------------------------------------------------------------------------------------------|
| [recipes](./recipes)              | Typical recipes for sparsifying ViT models along with any downloaded recipes from the SparseZoo.                      |
| tutorials            | Tutorial walkthroughs for how to sparsify ViT models using recipes (Coming Soon).                                                   |          |
| [README.md](./README.md)            | Readme file.                                                                                                        |
| [setup_integration.sh](./setup_integration.sh) | Setup file for the integration run from the command line.   

### Exporting for Inference

After sparsifying a model, to convert the model into an [ONNX](https://onnx.ai/) deployment format run the `export.py script`(https://github.com/neuralmagic/pytorch-image-models/blob/master/export.py).

The export process is modified such that the quantized and pruned models are corrected and folded properly.

For example, the following command can be run from within the integration's folder to export a trained/sparsified model's checkpoint:
```bash
python export.py \
    --checkpoint ./path/to/checkpoint/model.pth.tar \
    --recipe ../recipes/vit_base.85.recal.config.yaml \
    --save-dir ./exported-models \
    --filename vit_base_patch16_224 \
    --batch-size 1 \
    --image-shape 3 224 224 \
    --config ./path/to/checkpoint/args.yaml
```

The DeepSparse Engine [accepts ONNX formats](https://docs.neuralmagic.com/archive/sparseml/source/onnx_export.html) and is engineered to significantly speed up inference on CPUs for the sparsified models from this integration.
Examples for loading, benchmarking, and deploying can be found in the [DeepSparse repository here](https://github.com/neuralmagic/deepsparse).