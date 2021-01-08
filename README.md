<!---
Copyright 2021 Neuralmagic, Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# [Related Icon Here] SparseML

### Easy-to-use UI to optimize neural networks for better performance and smaller sizes
<p>
    <a href="https://github.com/neuralmagic/comingsoon/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/neuralmagic/comingsoon.svg?color=purple&style=for-the-badge" height=25>
    </a>
    <a href="https://docs.neuralmagic.com/sparseml/">
        <img alt="Documentation" src="https://img.shields.io/website/http/neuralmagic.com/sparseml/index.html.svg?down_color=red&down_message=offline&up_message=online&style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic/sparseml/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/sparseml.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://github.com/neuralmagic.com/comingsoon/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height=25>
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

Neural Magic, the creators of SparseML, is focused on making performance engineering for deep learning easy, affordable, and accessible. The Neural Magic Inference Engine enhances speed for neural networks in numerous ways, including activation sparsity and model pruning. SparseML is set up to allow easy creation of performance-optimized models, not just for Neural Magic but now for the ML performance community at large. Additionally, implementations for multiple frameworks, including PyTorch and TensorFlow, are provided. 

This repository includes model pruning APIs and CLIs as well as transfer learning APIs and CLIs, simplifying the process of achieving performance on deep learning models with Neural Magic.

### Repository Structure
```

neuralmagicML-python
    docs - API documentation for the repository
    neuralmagicML - The Python API Code
    notebooks - Tutorial Notebooks for using the Python API
    scripts - Functional scripts for working with the Python API
        onnx - Functional scripts for working with ONNX models
        pytorch - Functional scripts for working with PyTorch models
        server - Scripts to run the Sparsify server
        tensorflow - Functional scripts for working with TensorFlow models
    MANIFEST.in - Attaches server files to installation process
    README.md - readme file
    requirements.txt - requirements for the Python API
    setup.py - setuptools install script
```

## Quick Tour and Documentation
[TODO ENGINEERING: EDIT AS NEEDED]
Follow the quick tour below to get started.
For a more in-depth read, check out [SparseML documentation](https://docs.neuralmagic.com/sparseml/).

<!--- the docs url will become active once Marketing configures it. --->

### Installation and Requirements

- Installation of [SparseZoo](https://docs.neuralmagic.com/sparsezoo/) 
- Python 3.6.0 or higher
- Use Case: Computer Vision - Image Classification, Object Detection
- Model Architectures: Deep Learning Neural Network Architectures (e.g., CNNs, DNNs - refer to SparseZoo for examples)
- Instruction Set: Ideally CPUs with AVX-512 (e.g., Intel Xeon Cascade Lake, Icelake, Skylake) and 2 FMAs. VNNI support required for sparse quantization.
- OS / Environment: Ubuntu, CentOS, RHEL, Amazon Linux 
- FP32 bit precision (quantized performance coming soon)

To install, these packages will be required:

```python
$ pip install sparseml
$ pip install sparsezoo
```
General instructions for Python installation are found [here](https://realpython.com/installing-python/).

Optionally, you may also want to install the [Neural Magic Inference Engine](https://docs.neuralmagic.com/[ENGINE_REPO_NAME]/) and [Sparsify](https://docs.neuralmagic.com/sparsify/). SparseML can utilize Neural Magic’s runtime engine and Sparsify to achieve faster inference timings. For example, after running a benchmark, you might want to change optimization values and run an optimization profile again.

```python
$ pip install engine
$ pip install sparsify
```
Additionally, it is recommended to work within a virtual environment. 
Sample commands for creating and activating in a Unix-based system are provided below:
```
pip3 install virtualenv
python3 -m venv ./venv
source ./venv/bin/activate
```
1. Navigate to the parent directory of the `sparseml` codebase.
2. Use pip install to run the setup.py file in the repo: `pip install sparseml-python/`
3. Import sparseml library in your code: `import sparseml`

Note: If you run into issues with TensorFlow/PyTorch imports (specifically GPU vs. CPU support), 
you can edit the `requirements.txt` file at the root of the repository for the desired TensorFlow or PyTorch version.

## Usage

To use SparseML framework-specific tooling, the framework package(s) must already be installed in the environment.

The currently supported framework versions are:
- PyTorch supported versions: `>= 1.1.0, < 1.7.0`
- TensorFlow supported versions: >= `1.8.0` (TensorFlow >= `2.X` is not currently supported)

The following commands install versions of PyTorch and TensorFlow v1. For more installation information, see the [PyTorch](https://pytorch.org/get-started/locally/) and [TensorFlow](https://www.tensorflow.org/install/pip) websites.

```
pip3 install torch==1.5.0 torchvision==0.6.0
pip3 install tensorflow==1.15.0
```

## Tutorials
- SparseML Tutorials, which are implemented as Jupyter Notebooks for easy consumption and editing, are provided under the `notebooks` directory. 
- [SparseML Tutorials](notesbooks/) has more details on what's available and how to run them.

## Scripts
- Ease-of-use SparseML scripts, which are implemented as Python scripts for easy consumption and editing,
are provided under the `scripts` directory.
- [SparseML Scripts](scripts/) has more details on what's available and how to run them.
  
## Exporting to ONNX
- [ONNX](https://onnx.ai/) is a generic format for storing neural networks that is supported natively or by third-party extensions in all major deep learning frameworks such as PyTorch, TensorFlow, and Keras.
Due to this flexibility, ahould you use the Neural Magic Inference Engine, it uses the ONNX format.
- [Instructions for exporting to ONNX](docs/export-onnx.md) are available for these popular frameworks.

## Available Models and Recipes
If you are not ready to upload your model through SparseML, a number of pre-trained models are available in the [SparseZoo](https://docs.neuralmagic.com/sparsezoo/) that can be used. Included are both baseline and recalibrated models for higher performance. These can optionally be used with [Neural Magic Inference Engine](https://github.com/neuralmagic/engine/). The types available for each model architecture are noted in the [SparseML model repository listing](docs/available-models.md).

## Recalibration
APIs for recalibrating models are provided for each supported ML framework.
Recalibration includes
[model pruning (kernel sparsity)](https://towardsdatascience.com/pruning-deep-neural-network-56cae1ec5505) 
as well as [quantization](https://towardsdatascience.com/speeding-up-deep-learning-with-quantization-3fe3538cbb9)
in a future release.
Both of these, when paired with the Neural Magic Inference Engine, can significantly improve model inference speed.

The APIs are designed to be integrated into your existing code with as few lines as possible.
The implementations for each framework differ to best match their internal structures and designs.

To take advantage of these APIs, check out:

- ### [Configuration File Code Snippets](recal-configs.md)
- ### [Recalibrating in PyTorch](recal-python.md)
- ### [Recalibrating in TensorFlow](recal-tensorflow.md)

## Resources and Learning More
* [SparseML Documentation](https://docs.neuralmagic.com/sparseml/)
* [SparseML Use Cases](INSERT PATH HERE; IS THIS NEEDED?)
* [SparseML Examples](INSERT PATH HERE; IS THIS NEEDED?)
* [SparseZoo Documentation](https://docs.neuralmagic.com/sparsezoo/)
* [ENGINE_FORMAL_NAME Documentation](https://docs.neuralmagic.com/[ENGINE_REPO_NAME]/)
* [Neural Magic Blog](https://www.neuralmagic.com/blog/)
* [Neural Magic](https://www.neuralmagic.com/)

[TODO ENGINEERING: table with links for deeper topics or other links that should be included above]

## Contributing

We appreciate contributions to the code, documentation and examples, documentation!

- Report issues and bugs directly in [this GitHub project](https://github.com/neuralmagic/sparseml/issues).
- Learn how to work with the SparseML source code, including building and testing SparseML models and recipes as well as contributing code changes to SparseML by reading our [Development and Contribution guidelines](CONTRIBUTING.md).

Give SparseML a shout out on social! Are you able write a blog post, do a lunch ’n learn, host a meetup, or simply share via your networks? Help us build the community, yay! Here’s some details to assist:
- item 1 [TODO MARKETING: NEED METHODS]
- item n

## Join the Community

For user help or questions about SparseML, please use our [GitHub Discussions](https://www.github.com/neuralmagic/sparseml/issues). Everyone is welcome!

You can get the latest news, webinar invites, and other ML Performance tidbits by [connecting with the Neural Magic community](https://www.neuralmagic.com/NEED_URL/).[TODO MARKETING: NEED METHOD]

For more general questions about Neural Magic please contact us this way [Method](URL). [TODO MARKETING: NEED METHOD]

[TODO MARKETING: Example screenshot here]

## License

The project is licensed under the [Apache License Version 2.0](LICENSE).

## Release History

[Track this project via GitHub Releases.](https://github.com/neuralmagic/sparseml/releases)