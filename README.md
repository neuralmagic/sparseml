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

<div align="center">
    <h1><img alt="tool icon" src="https://raw.githubusercontent.com/neuralmagic/sparseml/main/docs/source/icon-sparseml.png" />&nbsp;&nbsp;SparseML</h1>

<h3>Libraries for training neural networks with sparsification recipes with a few lines of code, enabling faster and smaller models</h3>

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
</div>

SparseML is a toolkit for training neural networks with state-of-the-art [sparsification](https://docs.neuralmagic.com/main/source/getstarted.html#sparsification) algorithms such as pruning and quantization. SparseML allows you to integrate into leading deep learning libraries (e.g., Ultralytics, Hugging Face) and conducting sparse transfer learning onto leading deep learning models.

The [GitHub repository](https://github.com/neuralmagic/sparseml) contains integrations within the PyTorch, Keras, and TensorFlow V1 ecosystems, allowing for seamless model sparsification.

<!-- <img alt="SparseML Flow" src="https://docs.neuralmagic.com/docs/source/infographics/sparseml.png" width="960px" /> -->

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

This repository is tested on Python 3.6-3.9, and Linux/Debian systems. Using a virtual environment is highly recommended. Install the SparseML using the following command:

```bash
pip install sparseml
```

Supported deep learning frameworks: 
- `torch>=1.1.0,<=1.9.0`
- `tensorflow>=1.8.0,<=2.0.0`
- `tensorflow.keras >= 2.2.0`

More information on installation such as optional dependencies and requirements can be found [here.](https://docs.neuralmagic.com/sparseml/source/installation.html)

## 🏋️ Training

The following steps can be used for either the Computer Vision or NLP domain. The flow for interacting with our software is to first:

- Select a sparse pretrained model from the [SparseZoo](https://sparsezoo.neuralmagic.com/).
- Transfer learn the model on your custom dataset.

**OR**

- Download a recipe from the [SparseZoo](https://sparsezoo.neuralmagic.com/) associated with the model you are interested in sparsifying.
- Sparsify your model using the downloaded recipe.
- Transfer learn the model on your custom dataset.

### 🌱 Training a Sparse Pretrained Model | NLP Token Classification Example

To get started, install the PyTorch version of SparseML:

```bash
pip install sparseml[torch]
```

For NLP tranfer learning, you can select one of our sparse pretrained [models](https://sparsezoo.neuralmagic.com/?domain=nlp&sub_domain=masked_language_modeling&page=1) and its accompanying recipes. The good news is we already did plenty of experimentation to discover the appropriate parameters a particular recipe requires for transfer learning on a particular task. For example, if we were interested in the [BERT base 6 layer pruned 90%](https://sparsezoo.neuralmagic.com/models/nlp%2Fmasked_language_modeling%2Fbert-base%2Fpytorch%2Fhuggingface%2Fbookcorpus_wikitext%2F6layer_pruned90-none) model for the Named Entity Recognition (NER) task, we would run the following script:

```bash
sparseml.transformers.train.token_classification \
  --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/6layer_pruned90-none \
  --distill_teacher $MODEL_DIR/teacher \
  --dataset_name conll2003 \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --preprocessing_num_workers 16 \
  --output_dir $MODEL_DIR/6layer_pruned90-none \
  --fp16 \
  --seed 21097 \
  --num_train_epochs 5 \
  --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/6layer_pruned90-none?recipe_type=transfer-CoNLL2003 \
  --save_strategy epoch \
  --save_total_limit 2
```
For more info on CLI arguments run the following command and append the appropriate `task`:

```bash
sparseml.transformers.[task] --help
```

e.g. `sparseml.transformers.token_classification --help`

### 🌳 Export to ONNX

The SparseML installation additionally provided a `sparseml.transformers.export_onnx` command to convert your recently trained model to ONNX. Be sure the `--model_path` argument points to your trained model's location.

```bash
sparseml.transformers.export_onnx \
    --model_path "models/layer_pruned90-none" \
    --task 'token-classification' \
    --sequence_length 128   
```

More information on the codebase and contained processes can be found in the SparseML docs:
- [Sparsification Code](https://docs.neuralmagic.com/sparseml/source/code)
- [Sparsification Recipes](https://docs.neuralmagic.com/sparseml/source/recipes)
- [Exporting to ONNX](https://docs.neuralmagic.com/sparseml/source/onnx_export)

## Resources

#### Libraries
- [DeepSparse](https://docs.neuralmagic.com/deepsparse/)
- [SparseML](https://docs.neuralmagic.com/sparseml/)
- [SparseZoo](https://docs.neuralmagic.com/sparsezoo/)
- [Sparsify](https://docs.neuralmagic.com/sparsify/)

#### Versions

- [sparseml](https://pypi.org/project/sparseml/) | stable
- [sparseml-nightly](https://pypi.org/project/sparseml-nightly/) | nightly (dev)

#### Info

- [Blog](https://www.neuralmagic.com/blog/) 
- [Resources](https://www.neuralmagic.com/resources/)

Additionally, more information can be found via [GitHub Releases.](https://github.com/neuralmagic/sparseml/releases)

### License

The project is licensed under the [Apache License Version 2.0.](https://github.com/neuralmagic/sparseml/blob/main/LICENSE)

## Community

### Be Part of the Future... And the Future is Sparse!

Contribute with code, examples, integrations, and documentation as well as bug reports and feature requests! [Learn how here.](https://github.com/neuralmagic/sparseml/blob/main/CONTRIBUTING.md)

For user help or questions about SparseML, sign up or log in to our [**Deep Sparse Community Slack**](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). We are growing the community member by member and happy to see you there. Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue.](https://github.com/neuralmagic/sparseml/issues) You can get the latest news, webinar and event invites, research papers, and other ML Performance tidbits by [subscribing](https://neuralmagic.com/subscribe/) to the Neural Magic community.

For more general questions about Neural Magic, complete this [form.](http://neuralmagic.com/contact/)

### License

The project is licensed under the [Apache License Version 2.0.](https://github.com/neuralmagic/sparseml/blob/main/LICENSE)

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
