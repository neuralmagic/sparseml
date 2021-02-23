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
* model pruning
* quantization-aware-training
* sparse quantization-aware-training
* sparse transfer learning

## Installation
Both requirements can be installed via `pip` or can be cloned
and installed from their respective source repositories.

```bash
pip install git+https://github.com/rwightman/pytorch-image-models.git
pip install sparseml[torchvision]
```


## Script
`integrations/timm/train.py` modifies
[`train.py`](https://github.com/rwightman/pytorch-image-models/blob/master/train.py)
from pytorch-image-models to include a `sparseml-recipe` argument
to run SparseML optimizations with.  This can be a file path to a local
SparseML recipe or a SparseZoo model stub prefixed by `zoo:` such as
`zoo:cv-classification/resnet_v1-50/pytorch-rwightman/imagenet-augmented/pruned_quant-aggressive`.

Additionally, to run sparse transfer learning with a SparseZoo model that has
a transfer learning recipe, add `?recipe_type=transfer_learn` as part of the model stub.
i.e. `zoo:cv-classification/resnet_v1-50/pytorch-rwightman/imagenet-augmented/pruned_quant-aggressive?recipe_type=transfer_learn`.
This will run a recipe that holds the optimized sparsity structure the same while allowing
non-zero weights to be updated during training, so pre-learned optimizations can be applied
to different datasets.

To load the base weights for a SparseZoo recipe as the initial checkpoint, set
`--initial-checkpoint` to `zoo`.  To use the weights of a SparseZoo model as the
initial checkpoint, pass that model's SparseZoo stub prefixed by `zoo:` to the
`--initial-checkpoint` argument.

Running the script will
follow the normal pytorch-image-models training flow with the given
SparseML optimizations enabled.

Some considerations:

* `--sparseml-recipe` is a required parameter
* `--epochs` will now be overridden by the epochs set in the SparseML recipe
* Modifiers will log their outputs to the console as well as to a tensorboard file
* After training is complete, the final model will be exported to ONNX using SparseML

You can learn how to build or download a recipe using the
[SparseML](https://github.com/neuralmagic/sparseml)
or [SparseZoo](https://github.com/neuralmagic/sparsezoo)
documentation, or export one with [Sparsify](https://github.com/neuralmagic/sparsify).

Documentation on the original script can be found
[here](https://rwightman.github.io/pytorch-image-models/scripts/).
The latest commit hash that `train.py` is based on is included in the docstring.


#### Example Command
Training from a local recipe and checkpoint
```bash
python integrations/timm/train.py \
  /PATH/TO/DATASET/imagenet/ \
  --sparseml-recipe /PATH/TO/RECIPE/recipe.yaml \
  --initial-checkpoint PATH/TO/CHECKPOINT/model.pth \
  --dataset imagenet \
  --batch-size 64 \
  --remode pixel --reprob 0.6 --smoothing 0.1 \
  --output models/optimized \
  --model resnet50 \
  --workers 8 \
```  

Training from a local recipe and SparseZoo checkpoint
```bash
python integrations/timm/train.py \
  /PATH/TO/DATASET/imagenet/ \
  --sparseml-recipe /PATH/TO/RECIPE/recipe.yaml \
  --initial-checkpoint zoo:model/stub/path \
  --dataset imagenet \
  --batch-size 64 \
  --remode pixel --reprob 0.6 --smoothing 0.1 \
  --output models/optimized \
  --model resnet50 \
  --workers 8 \
```  

Training from a SparseZoo recipe and checkpoint with sparse transfer learning enabled
```bash
python integrations/timm/train.py \
  /PATH/TO/DATASET/imagenet/ \
  --sparseml-recipe zoo:model/stub/path?recipe_type=transfer_learn \
  --initial-checkpoint zoo \
  --dataset imagenet \
  --batch-size 64 \
  --remode pixel --reprob 0.6 --smoothing 0.1 \
  --output models/optimized \
  --model resnet50 \
  --workers 8 \
```  
