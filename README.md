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

<h1><img alt="tool icon" src="https://raw.githubusercontent.com/neuralmagic/sparseml/main/docs/source/icon-sparseml.png" />&nbsp;&nbsp;SparseML</h1>

<h3>Libraries for applying sparsification recipes to neural networks with a few lines of code, enabling faster and smaller models</h3>

<p>
    <a href="https://docs.neuralmagic.com/sparseml/">
        <img alt="Documentation" src="https://img.shields.io/badge/documentation-darkred?&style=for-the-badge&logo=read-the-docs" height=25>
    </a>
    <a href="https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ/">
        <img src="https://img.shields.io/badge/slack-purple?style=for-the-badge&logo=slack" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparseml/issues">
        <img src="https://img.shields.io/badge/support%20forums-navy?style=for-the-badge&logo=github" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparseml/actions/workflows/test-check.yaml">
        <img alt="Main" src="https://img.shields.io/github/workflow/status/neuralmagic/sparseml/Test%20Checks/main?label=build&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparseml/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/sparseml.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparseml/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/neuralmagic/sparseml.svg?color=lightgray&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparseml/blob/main/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height=25>
    </a>
    <a href="https://www.youtube.com/channel/UCo8dO_WMGYbWCRnj_Dxr4EA">
        <img src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height=25>
    </a>
     <a href="https://medium.com/limitlessai">
        <img src="https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white" height=25>
    </a>
    <a href="https://twitter.com/neuralmagic">
        <img src="https://img.shields.io/twitter/follow/neuralmagic?color=darkgreen&label=Follow&style=social" height=25>
    </a>
</p>

## Overview

SparseML is a toolkit that includes APIs, CLIs, scripts and libraries that apply state-of-the-art [sparsification](https://docs.neuralmagic.com/main/source/getstarted.html#sparsification) algorithms such as pruning and quantization to any neural network. 
General, recipe-driven approaches built around these algorithms enable the simplification of creating faster and smaller models for the ML performance community at large.

The [GitHub repository](https://github.com/neuralmagic/sparseml) contains integrations within the PyTorch, Keras, and TensorFlow V1 ecosystems, allowing for seamless model sparsification.

<img alt="SparseML Flow" src="https://docs.neuralmagic.com/docs/source/infographics/sparseml.png" width="960px" />

## Highlights

### Integrations

<p>
    <a href="https://github.com/neuralmagic/sparseml/tree/main/integrations/pytorch">
        <img src="https://docs.neuralmagic.com/docs/source/highlights/sparseml/pytorch-torchvision.png" width="136px" />
    </a>
    <a href="https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov3">
        <img src="https://docs.neuralmagic.com/docs/source/highlights/sparseml/ultralytics-yolov3.png" width="136px" />
    </a>
    <a href="https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov5">
        <img src="https://docs.neuralmagic.com/docs/source/highlights/sparseml/ultralytics-yolov5.png" width="136px" />
    </a>
    <a href="https://github.com/neuralmagic/sparseml/tree/main/integrations/huggingface-transformers">
        <img src="https://docs.neuralmagic.com/docs/source/highlights/sparseml/huggingface-transformers.png" width="136px" />
    </a>
    <a href="https://github.com/neuralmagic/sparseml/tree/main/integrations/rwightman-timm">
        <img src="https://docs.neuralmagic.com/docs/source/highlights/sparseml/rwightman-timm.png" width="136px" />
    </a>
</p>

### Creating Sparse Models

<p>
    <a href="https://github.com/neuralmagic/sparseml/tree/main/integrations/pytorch/notebooks/classification.ipynb">
        <img src="https://docs.neuralmagic.com/docs/source/tutorials/classification_resnet-50.png" width="136px" />
    </a>
    <a href="https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov3/tutorials/sparsifying_yolov3_using_recipes.md">
        <img src="https://docs.neuralmagic.com/docs/source/tutorials/detection_yolov3.png" width="136px" />
    </a>
    <a href="https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov5/tutorials/sparsifying_yolov5_using_recipes.md">
        <img src="https://docs.neuralmagic.com/docs/source/tutorials/detection_yolov5.png" width="136px" />
    </a>
    <a href="https://github.com/neuralmagic/sparseml/tree/main/integrations/huggingface-transformers/tutorials/sparsifying_bert_using_recipes.md">
        <img src="https://docs.neuralmagic.com/docs/source/tutorials/nlp_bert.png" width="136px" />
    </a>
</p>

### Transfer Learning from Sparse Models

<p>
    <a href="https://github.com/neuralmagic/sparseml/tree/main/integrations/pytorch/notebooks/sparse_quantized_transfer_learning.ipynb">
        <img src="https://docs.neuralmagic.com/docs/source/tutorials/classification_resnet-50.png" width="136px" />
    </a>
    <a href="https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/tutorials/yolov3_sparse_transfer_learning.md">
        <img src="https://docs.neuralmagic.com/docs/source/tutorials/detection_yolov3.png" width="136px" />
    </a>
    <a href="https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/tutorials/yolov5_sparse_transfer_learning.md">
        <img src="https://docs.neuralmagic.com/docs/source/tutorials/detection_yolov5.png" width="136px" />
    </a>
</p>


## Tutorials

### 🖼️ Computer Vision

- [Sparsifying PyTorch Models Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/pytorch/tutorials/sparsifying_pytorch_models_using_recipes.md)
- [Sparsifying YOLOv3 Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/tutorials/sparsifying_yolov3_using_recipes.md)
- [Sparsifying YOLOv5 Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/tutorials/sparsifying_yolov5_using_recipes.md)
- [Sparsifying YOLACT Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/dbolya-yolact/tutorials/sparsifying_yolact_using_recipes.md)
- [Sparse Transfer Learning for Image Classification](https://github.com/neuralmagic/sparseml/blob/main/integrations/pytorch/tutorials/classification_sparse_transfer_learning_tutorial.md)
- [Sparse Transfer Learning With YOLOv3](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov3/tutorials/yolov3_sparse_transfer_learning.md)
- [Sparse Transfer Learning With YOLOv5](https://github.com/neuralmagic/sparseml/blob/main/integrations/ultralytics-yolov5/tutorials/yolov5_sparse_transfer_learning.md)

&emsp; **Notebooks**

- [Keras Image Classification Model Pruning Using SparseML](https://github.com/neuralmagic/sparseml/blob/main/integrations/keras/notebooks/classification.ipynb)
- [PyTorch Image Classification Model Pruning Using SparseML](https://github.com/neuralmagic/sparseml/blob/main/integrations/pytorch/notebooks/classification.ipynb)
- [PyTorch Image Detection Model Pruning Using SparseML](https://github.com/neuralmagic/sparseml/blob/main/integrations/pytorch/notebooks/detection.ipynb)
- [Sparse-Quantized Transfer Learning in PyTorch Using SparseML](https://github.com/neuralmagic/sparseml/blob/main/integrations/pytorch/notebooks/sparse_quantized_transfer_learning.ipynb)
- [Torchvision Classification Model Pruning Using SparseML](https://github.com/neuralmagic/sparseml/blob/main/integrations/pytorch/notebooks/torchvision.ipynb)
- [TensorFlow v1 Classification Model Pruning Using SparseML](https://github.com/neuralmagic/sparseml/blob/main/integrations/tensorflow_v1/notebooks/classification.ipynb)

### 📰 NLP
- [Sparsifying BERT Models Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/tutorials/sparsifying_bert_using_recipes.md)
- [Sparse Transfer Learning With BERT](https://github.com/neuralmagic/sparseml/blob/main/integrations/huggingface-transformers/tutorials/bert_sparse_transfer_learning.md)


## Installation

This repository is tested on Python 3.7-3.9, and Linux/Debian systems.
It is recommended to install in a [virtual environment](https://docs.python.org/3/library/venv.html) to keep your system in order.
Currently supported ML Frameworks are the following: `torch>=1.1.0,<=1.9.0`, `tensorflow>=1.8.0,<=2.0.0`, `tensorflow.keras >= 2.2.0`.

Install with pip using:

```bash
pip install sparseml
```

More information on installation such as optional dependencies and requirements can be found [here.](https://docs.neuralmagic.com/sparseml/source/installation.html)

## Quick Tour

To enable flexibility, ease of use, and repeatability, sparsifying a model is done using a recipe.
The recipes encode the instructions needed for modifying the model and/or training process as a list of modifiers.
Example modifiers can be anything from setting the learning rate for the optimizer to gradual magnitude pruning.
The files are written in [YAML](https://yaml.org/) and stored in YAML or [markdown](https://www.markdownguide.org/) files using [YAML front matter.](https://assemble.io/docs/YAML-front-matter.html) The rest of the SparseML system is coded to parse the recipes into a native format for the desired framework and apply the modifications to the model and training pipeline.

`ScheduledModifierManager` classes can be created from recipes in all supported ML frameworks.
The manager classes handle overriding the training graphs to apply the modifiers as described in the desired recipe.
Managers can apply recipes in one shot or training aware ways. 
One shot is invoked by calling `.apply(...)` on the manager while training aware requires calls into `initialize(...)` (optional), `modify(...)`, and `finalize(...)`.

For the frameworks, this means only a few lines of code need to be added to begin supporting pruning, quantization, and other modifications to most training pipelines.
For example, the following applies a recipe in a training aware manner:
```python
model = Model()  # model definition
optimizer = Optimizer()  # optimizer definition
train_data = TrainData()  # train data definition
batch_size = BATCH_SIZE  # training batch size
steps_per_epoch = len(train_data) // batch_size

from sparseml.pytorch.optim import ScheduledModifierManager
manager = ScheduledModifierManager.from_yaml(PATH_TO_RECIPE)
optimizer = manager.modify(model, optimizer, steps_per_epoch)

# PyTorch training code

manager.finalize(model)
```

Instead of training aware, the following example code shows how to execute a recipe in a one shot manner:
```python
model = Model()  # model definition

from sparseml.pytorch.optim import ScheduledModifierManager
manager = ScheduledModifierManager.from_yaml(PATH_TO_RECIPE)
manager.apply(model)
```

More information on the codebase and contained processes can be found in the SparseML docs:
- [Sparsification Code](https://docs.neuralmagic.com/sparseml/source/code)
- [Sparsification Recipes](https://docs.neuralmagic.com/sparseml/source/recipes)
- [Exporting to ONNX](https://docs.neuralmagic.com/sparseml/source/onnx_export)

## Resources

### Learning More

- Documentation: [SparseML,](https://docs.neuralmagic.com/sparseml/) [SparseZoo,](https://docs.neuralmagic.com/sparsezoo/) [Sparsify,](https://docs.neuralmagic.com/sparsify/) [DeepSparse](https://docs.neuralmagic.com/deepsparse/)
- Neural Magic: [Blog,](https://www.neuralmagic.com/blog/) [Resources](https://www.neuralmagic.com/resources/)

### Release History

Official builds are hosted on PyPI

- stable: [sparseml](https://pypi.org/project/sparseml/)
- nightly (dev): [sparseml-nightly](https://pypi.org/project/sparseml-nightly/)

Additionally, more information can be found via [GitHub Releases.](https://github.com/neuralmagic/sparseml/releases)

### License

The project is licensed under the [Apache License Version 2.0.](https://github.com/neuralmagic/sparseml/blob/main/LICENSE)

## Community

### Contribute

We appreciate contributions to the code, examples, integrations, and documentation as well as bug reports and feature requests! [Learn how here.](https://github.com/neuralmagic/sparseml/blob/main/CONTRIBUTING.md)

### Join

For user help or questions about SparseML, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). We are growing the community member by member and happy to see you there. Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues)

You can get the latest news, webinar and event invites, research papers, and other ML Performance tidbits by [subscribing](https://neuralmagic.com/subscribe/) to the Neural Magic community.

For more general questions about Neural Magic, please fill out this [form.](http://neuralmagic.com/contact/)

### Cite

Find this project useful in your research or other communications? Please consider citing:

```bibtex
@InProceedings{
    pmlr-v119-kurtz20a, 
    title = {Inducing and Exploiting Activation Sparsity for Fast Inference on Deep Neural Networks}, 
    author = {Kurtz, Mark and Kopinsky, Justin and Gelashvili, Rati and Matveev, Alexander and Carr, John and Goin, Michael and Leiserson, William and Moore, Sage and Nell, Bill and Shavit, Nir and Alistarh, Dan}, 
    booktitle = {Proceedings of the 37th International Conference on Machine Learning}, 
    pages = {5533--5543}, 
    year = {2020}, 
    editor = {Hal Daumé III and Aarti Singh}, 
    volume = {119}, 
    series = {Proceedings of Machine Learning Research}, 
    address = {Virtual}, 
    month = {13--18 Jul}, 
    publisher = {PMLR}, 
    pdf = {http://proceedings.mlr.press/v119/kurtz20a/kurtz20a.pdf},
    url = {http://proceedings.mlr.press/v119/kurtz20a.html}, 
    abstract = {Optimizing convolutional neural networks for fast inference has recently become an extremely active area of research. One of the go-to solutions in this context is weight pruning, which aims to reduce computational and memory footprint by removing large subsets of the connections in a neural network. Surprisingly, much less attention has been given to exploiting sparsity in the activation maps, which tend to be naturally sparse in many settings thanks to the structure of rectified linear (ReLU) activation functions. In this paper, we present an in-depth analysis of methods for maximizing the sparsity of the activations in a trained neural network, and show that, when coupled with an efficient sparse-input convolution algorithm, we can leverage this sparsity for significant performance gains. To induce highly sparse activation maps without accuracy loss, we introduce a new regularization technique, coupled with a new threshold-based sparsification method based on a parameterized activation function called Forced-Activation-Threshold Rectified Linear Unit (FATReLU). We examine the impact of our methods on popular image classification models, showing that most architectures can adapt to significantly sparser activation maps without any accuracy loss. Our second contribution is showing that these these compression gains can be translated into inference speedups: we provide a new algorithm to enable fast convolution operations over networks with sparse activations, and show that it can enable significant speedups for end-to-end inference on a range of popular models on the large-scale ImageNet image classification task on modern Intel CPUs, with little or no retraining cost.} 
}
```

```bibtex
@misc{
    singh2020woodfisher,
    title={WoodFisher: Efficient Second-Order Approximation for Neural Network Compression}, 
    author={Sidak Pal Singh and Dan Alistarh},
    year={2020},
    eprint={2004.14340},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
