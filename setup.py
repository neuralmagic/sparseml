# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from typing import Dict, List, Tuple

from setuptools import find_packages, setup


_NIGHTLY = "nightly" in sys.argv
if _NIGHTLY:
    # remove nightly param so it does not break bdist_wheel
    sys.argv.remove("nightly")


_PACKAGE_NAME = "sparseml" if not _NIGHTLY else "sparseml-nightly"
_VERSION = "0.1.0"


_deps = [
    "jupyter>=1.0.0",
    "ipywidgets>=7.0.0",
    "pyyaml>=5.0.0",
    "progressbar2>=3.0.0",
    "numpy>=1.0.0",
    "matplotlib>=3.0.0",
    "merge-args>=0.1.0",
    "onnx>=1.5.0,<1.8.0",
    "onnxruntime>=1.0.0",
    "pandas<1.0.0",
    "psutil>=5.0.0",
    "requests>=2.0.0",
    "scikit-image>=0.15.0",
    "scipy>=1.0.0",
    "tqdm>=4.0.0",
    "toposort>=1.0",
]
_nm_deps = [
    f"sparsezoo~={_VERSION}" if not _NIGHTLY else "sparsezoo-nightly",
]
_pytorch_deps = ["torch>=1.1.0", "tensorboard>=1.0", "tensorboardX>=1.0"]
_pytorch_vision_deps = _pytorch_deps + ["torchvision>=0.3.0"]
_tensorflow_v1_deps = ["tensorflow<2.0.0", "tensorboard<2.0.0", "tf2onnx>=1.0.0,<1.6"]
_tensorflow_v1_gpu_deps = [
    "tensorflow-gpu<2.0.0",
    "tensorboard<2.0.0",
    "tf2onnx>=1.0.0,<1.6",
]
_keras_deps = ["tensorflow~=2.2.0", "keras2onnx>=1.0.0"]

_dev_deps = [
    "black>=20.8b1",
    "flake8>=3.8.3",
    "isort>=5.7.0",
    "rinohtype>=0.4.2",
    "recommonmark>=0.7.0",
    "sphinx>=3.4.0",
    "sphinx-copybutton>=0.3.0",
    "sphinx-markdown-tables>=0.0.15",
    "wheel>=0.36.2",
    "pytest>=6.0.0",
    "sphinx-rtd-theme",
]


def _setup_packages() -> List:
    return find_packages(
        "src", include=["sparseml", "sparseml.*"], exclude=["*.__pycache__.*"]
    )


def _setup_package_dir() -> Dict:
    return {"": "src"}


def _setup_install_requires() -> List:
    return _nm_deps + _deps


def _setup_extras() -> Dict:
    return {
        "dev": _dev_deps,
        "torch": _pytorch_deps,
        "torchvision": _pytorch_vision_deps,
        "tf_v1": _tensorflow_v1_deps,
        "tf_v1_gpu": _tensorflow_v1_gpu_deps,
        "tf_keras": _keras_deps,
    }


def _setup_entry_points() -> Dict:
    return {}


def _setup_long_description() -> Tuple[str, str]:
    return open("README.md", "r", encoding="utf-8").read(), "text/markdown"


setup(
    name=_PACKAGE_NAME,
    version=_VERSION,
    author="Neuralmagic, Inc.",
    author_email="support@neuralmagic.com",
    description=(
        "Libraries for state-of-the-art deep neural network optimization algorithms, "
        "enabling simple pipelines integration with a few lines of code"
    ),
    long_description=_setup_long_description()[0],
    long_description_content_type=_setup_long_description()[1],
    keywords=(
        "inference, machine learning, neural network, computer vision, nlp, cv, "
        "deep learning, torch, pytorch, tensorflow, keras, sparsity, pruning, "
        "deep learning libraries, onnx, quantization, automl"
    ),
    license="Apache",
    url="https://github.com/neuralmagic/sparseml",
    package_dir=_setup_package_dir(),
    packages=_setup_packages(),
    install_requires=_setup_install_requires(),
    extras_require=_setup_extras(),
    entry_points=_setup_entry_points(),
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
