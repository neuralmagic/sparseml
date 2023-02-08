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

### FFCV SparseML support

[FFCV] or Fast Forward Computer Vision is a drop-in data-loading system that 
can dramatically increase data throughput in model training. [SparseML] now 
provides experimental support for FFCV with it's [ImageNet] based datasets.

### Installation

Use conda to install all ffcv dependencies; install conda using instructions from
[conda.io](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda create -n ffcv python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge && conda activate ffcv
pip install ffcv
pip install sparseml
```

### Usage

To use [FFCV] with [SparseML], just pass in the `--ffcv` flag to training script.
Example command:
```bash
sparseml.image_classification.train \
  --recipe-path "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none?recipe_type=original" \
  --arch-key resnet50 --checkpoint-path zoo \
  --dataset imagenette --train-batch-size 128 \
  --test-batch-size 256 --dataset-path dataset \
  --loader-num-workers 8 --ffcv
```



[FFCV]: https://ffcv.io/
[SparseML]: https://github.com/neuralmagic/sparseml
[ImageNet]: https://www.image-net.org/