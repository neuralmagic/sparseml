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

# SparseML-ultralytics/yolov5 integration
This directory provides a SparseML integrated training script for the popular
[ultralytics/yolov5](https://github.com/ultralytics/yolov5)
repository also known as [timm](https://pypi.org/project/timm/).

Using this integration, you will be able to apply SparseML optimizations
to the powerful training flows provided in the yolov5 repository.

Some of the tasks you can perform using this integration include, but are not limited to:
* model pruning
* quantization-aware-training
* sparse quantization-aware-training
* sparse transfer learning

## Installation
To use both the script, clone both repositories, install their dependencies,
and copy the integrated training script into the yolov5 directory to run from.

```bash
# clone
git clone https://github.com/ultralytics/yolov5.git
git clone https://github.com/neuralmagic/sparseml.git

# copy script
cp sparseml/examples/ultralytics-sparseml/main.py yolov5
cd yolov5

# install dependencies
pip install -r requirements.txt
pip install sparseml
```


## Script
`examples/timm-sparseml/main.py` modifies
[`train.py`](https://github.com/ultralytics/yolov5/blob/master/train.py)
from yolov5 to include a `sparseml-recipe-path` argument
to run SparseML optimizations with.  This can be a file path to a local
SparseML recipe or a SparseZoo model stub prefixed by `zoo:` such as
`zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned-aggressive`.

Additionally, for sparse transfer learning, the flag `--sparse-transfer-learn`
was added.  Running the script with this flag will add modifiers to the given
recipe that will keep the base sparsity constant during training, allowing
the model to learn the new dataset while keeping the same optimized structure.
If a SparseZoo recipe path is provided with sparse transfer learning enabled,
then the the model's specific "transfer" recipe will be loaded instead.

To load the base weights for a SparseZoo recipe as the initial checkpoint, set
`--initial-checkpoint` to `zoo`.  To use the weights of a SparseZoo model as the
initial checkpoint, pass that model's SparseZoo stub prefixed by `zoo:` to the
`--initial-checkpoint` argument.

Running the script will
follow the normal yolov5 training flow with the given SparseML optimizations enabled.

Some considerations:

* `--sparseml-recipe-path` is a required parameter
* `--epochs` will now be overridden by the epochs set in the SparseML recipe
* if using learning rate schedulers both with the yolov5 script and your recipe, they
may conflict with each other causing unintended side effects, choose
hyperparameters accordingly.
* Modifiers will log their outputs to the console as well as to the tensorboard file
* After training is complete, the final model will be exported to ONNX using SparseML

You can learn how to build or download a recipe using the
[SparseML](https://github.com/neuralmagic/sparseml)
or [SparseZoo](https://github.com/neuralmagic/sparsezoo)
documentation, or export one with [Sparsify](https://github.com/neuralmagic/sparsify).

Documentation on the original script can be found
[here](https://github.com/ultralytics/yolov5).
The latest commit hash that `main.py` is based on is included in the docstring.


#### Example Command
Call the script from the `yolov5` directory, passing in the same arguments as
`train.py`, with the additional SparseML argument(s) included.
```bash
python main.py \
  --sparseml-recipe-path /PATH/TO/RECIPE/recipe.yaml \
  <regular yolov5/train.py paramters>
```  
