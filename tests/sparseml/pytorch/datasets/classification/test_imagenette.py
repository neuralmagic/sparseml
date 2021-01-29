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

from sparseml.pytorch.datasets import (
    DatasetRegistry,
    ImagenetteDataset,
    ImagewoofDataset,
)


def _validate(dataset: Dataset, size: int):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 3
    assert item[0].shape[1] == size
    assert item[0].shape[2] == size
    assert item[1] < 10


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False),
    reason="Skipping dataset tests",
)
def test_imagenette_160():
    train_dataset = ImagenetteDataset(train=True)
    _validate(train_dataset, 160)

    val_dataset = ImagenetteDataset(train=False)
    _validate(val_dataset, 160)

    reg_dataset = DatasetRegistry.create("imagenette", train=False)
    _validate(reg_dataset, 160)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False),
    reason="Skipping dataset tests",
)
def test_imagewoof_160():
    train_dataset = ImagewoofDataset(train=True)
    _validate(train_dataset, 160)

    val_dataset = ImagewoofDataset(train=False)
    _validate(val_dataset, 160)

    reg_dataset = DatasetRegistry.create("imagewoof", train=False)
    _validate(reg_dataset, 160)
