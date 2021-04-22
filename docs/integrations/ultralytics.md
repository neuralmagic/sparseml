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

# Ultralytics/yolov5-SparseML Integration
This document explains how to use SparseML directly with the
[ultralytics/yolov5](https://github.com/ultralytics/yolov5) repository to sparsify
(prune and quantize) YOLO models.

SparseML integrates directly into yolov5's `train.py` script by adding the additional
`--sparseml-recipe` argument.  Users can either pass in the file path to a local recipe
file or a [SparseZoo](https://github.com/neuralmagic/sparsezoo) stub of a pre-sparsified
model to use that model's recipe.  Learn more about SparseML recipes
[here](https://docs.neuralmagic.com/sparseml/source/recipes.html).

### SparseZoo Stubs
The following SparseZoo stubs can be used directly with `--sparseml-recipes` to load
their associated recipes

| Model | Description | SparseZoo Stub |
| ----------- | ----------- | ----------- |
| YOLOv3-pruned | 88% sparse YOLOv3-SPP model | "zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned-aggressive_97" |
| YOLOv3-pruned_quant | 83% sparse YOLOv3-SPP model with INT8 quantization | "zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94" |


### Other Arguments
When a recipe is passed in to `--sparseml-recipe`, a `ScheduledModifierManager`
and `ScheduledOptimizer` will be created to modify the normal training flow for
sparsification. For best results with SparseML the following arguments may also
have to be set.  In general, all should be set when running with SparseML
(with the exception of `--disable-amp` if not using quantization).

* `--export-onnx` - Exports the final model to ONNX after training completes. Required
    for inference with the [DeepSparse](https://github.com/neuralmagic/deepsparse)
    engine
* `--use-leaky-relu` - LeakyReLU runs best on the DeepSparse Engine. Setting this flag
    overrides the default SiLU activations with LeakyReLU. If using pre-trained weights
    be sure that the pre-trained model also used LeakyReLU activations
* `--disable-ema` - Exponential moving average (EMA) is not currently supported when
    pruning. This argument disables its usage
* `--disable-amp` - PyTorch quantization-aware-training is currently incompatible with
    mixed-precision (`torch.cuda.amp`). This flag disables mixed-precision
    

### Base Weights
Base weights for a `YOLOv3-SPP` model with LeakyReLU activations can be downloaded from
SparseZoo as follows.  The model achieves 64.2 mAP@0.5.

```python
from sparsezoo.models.detection import yolo_v3

base_model = yolo_v3()
print(f"downloaded path: {base_model.framework_files[0].downloaded_path()}")
```

After downloading, the weights can be moved to a file location of your choice:
```bash
cp <DOWNLOADED-PATH> <YOUR-LOCAL-PATH>
```

### Example command
```bash
python train.py \
  --sparseml-recipe zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94 \
  --use-leaky-relu \
  --weights <YOUR-LOCAL-YOLOv3-SPP-BASE-PATH> \
  --export-onnx \
  --disable-ema \
  --disable-amp \
  --cfg ./models/hub/yolov3-spp.yaml \
  --data coco.yaml \
  --hyp data/hyp.finetune.yaml \
  --epochs 242 \
  --batch-size 48 \
  --name yolov3-spp-leaky_relu-pruned_quant
```

### Benchmarking and Examples
For examples of benchmarking and deploying sparsified YOLO models with the DeepSparse
engine, take a look at the `deepsparse/` examples
[here](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics).

### Issues
Please direct any issues with the SparseML YOLO integration to the [SparseML repo issues
page](https://github.com/neuralmagic/sparseml/issues).
