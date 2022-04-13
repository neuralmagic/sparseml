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

## Training

The following steps can be used for either the Computer Vision and NLP domains. The sections below will observe an example of using SparseML in the NLP domain for the token classification task (NER) even though you can apply any of these steps across domains and tasks.

### Step 1: Obtaining a Dense Teacher Model

Before applying sparse transfer learning, a dense teacher model must first be included in the process in order to distill its "knowledge" to a student model. You have two options for obtaining a dense teacher:

- Train your own teacher model.
- Download an available teacher model from the SparseZoo. For NLP, please take a look at the select list of [masked language models](https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=masked_language_modeling) and for Computer Vision please see [placeholder]

If you prefer to train your own dense teacher model, the following is an example of using a script for training BERT for the token classification task (NER):

```bash
sparseml.transformers.token_classification \
    --output_dir models/teacher \
    --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none \
    --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/base-none?recipe_type=transfer-token_classification \
    --recipe_args '{"init_lr":0.00003}' \
    --dataset_name conll2003 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --preprocessing_num_workers 6 \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --fp16 \
    --save_strategy epoch \
    --save_total_limit 1
```

The training command should run to completion in less than a few hours. Once the command has completed, you will have a deployable sparse model located in `models/teacher`.

### Step 2: Distilling Teacher Model to Student Model

Now that we have a dense teacher model, we'll use the 80% sparse-quantized BERT model from the SparseZoo and fine-tune it on the CoNLL-2003 dataset, resulting in a model that achieves 98.59% F1 on the validation set. Keep in mind that the `--distill_teacher` argument is set to pull a dense CoNLL-2003 model from the SparseZoo to enable it to run independent of the dense teacher step above.

```bash
sparseml.transformers.train.token_classification \
    --output_dir models/sparse_quantized \
    --model_name_or_path zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni \
    --recipe zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni?recipe_type=transfer-token_classification \
    --recipe_args '{"init_lr":0.00005}' \
    --distill_teacher zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/base-none \
    --dataset_name conll2003 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --preprocessing_num_workers 6 \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --fp16 \
    --save_strategy epoch \
    --save_total_limit 1
```

### STEP 3: Export Student Model to ONNX

The SparseML installation additionally provided a sparseml.transformers.export_onnx command. You will use this to load the training model folder and create a new model.onnx file within. Be sure the --model_path argument points to your trained model. By default, it is set to the result from transfer learning a sparse-quantized BERT model: --model_path "models/sparse_quantized".

```bash
sparseml.transformers.export_onnx \
    --model_path "models/sparse_quantized" \
    --task 'token-classification' \
    --sequence_length 128   
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
