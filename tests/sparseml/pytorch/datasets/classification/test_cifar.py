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

import os

import pytest
from torch.utils.data import Dataset

from sparseml.pytorch.datasets import CIFAR10Dataset, CIFAR100Dataset, DatasetRegistry


def _validate_cifar(dataset: Dataset, num_classes: int):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 3
    assert item[0].shape[1] == 32
    assert item[0].shape[2] == 32
    assert item[1] < num_classes


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False),
    reason="Skipping dataset tests",
)
def test_cifar_10():
    train_dataset = CIFAR10Dataset(train=True)
    _validate_cifar(train_dataset, 10)

    val_dataset = CIFAR10Dataset(train=False)
    _validate_cifar(val_dataset, 10)

    reg_dataset = DatasetRegistry.create("cifar10", train=False)
    _validate_cifar(reg_dataset, 10)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False),
    reason="Skipping dataset tests",
)
def test_cifar_100():
    train_dataset = CIFAR100Dataset(train=True)
    _validate_cifar(train_dataset, 100)

    val_dataset = CIFAR100Dataset(train=False)
    _validate_cifar(val_dataset, 100)

    reg_dataset = DatasetRegistry.create("cifar100", train=False)
    _validate_cifar(reg_dataset, 100)
