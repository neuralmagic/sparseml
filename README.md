# Neuralmagic Pytorch project

At Neuralmagic we offer the ability to run at GPU speeds directly on CPU through algorithmic innovation in the space of SW performance engineering, Machine Learning and Systems expertise.
This repository is complementary to the Neuralmagic engine package, and is designed with the Machine Learning engineer in mind. 
Specifically, this repo covers basic and advanced concepts which allow performance acceleration through the Neuralmagic engine.

## Tutorials organization

tutorials are split into jupyter notebooks which are self contained and provide detailed walkthroughs of the following topics:

* `model_training.ipynb` - seting up model / data and training
* `kernel_sparsity_pruning.ipynb` - sparsity, why is it useful, how to achieve it - i.e. pruning
* `activation_sparsity.ipynb` - feature (activation) sparsity origin and assessment

together these notebooks are aimed at providing a step by step guide for the evaluation and preparation of ones models such that they may be accelerated by the Neuralmagic engine package.

## Prerequisites and installation

set up your environment, for example using venv:


```
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```

install requirements from requirements.txt:

```
pip install -r requirements.txt
```
In addition we require the installation of:

```
torch==1.1.0
torchvision==0.3.0
```

The installation of `torch` and `torchvision` depends on your machine and specific cuda driver. 
please refer to https://pytorch.org/get-started/locally/ for installation details.



## Evaluation License Agreement
This project and repo is proprietary Neuralmagic intellectual property and is made available strictly privately and under NDA on a per user/organization basis.
By using this repo the user (organization) accepts the terms and conditions of the attached [evaluation license agreement](https://bitbucket.org/neuralmagic/neuralmagicml-pytorch/src/master/Evaluation%20SLA%20(Neuralmagic)%20v2.pdf).

## Third party licenses

* Packages as defined in the installed requirements above and their associated licenses;
* Examples given in the notebooks make use of fast.ai's [Imagenette dataset](https://github.com/fastai/imagenette) provided under the [Apache License 2.0](https://github.com/fastai/imagenette/blob/master/LICENSE)
