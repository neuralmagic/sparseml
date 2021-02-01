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
import torch
from torch.utils.data import Dataset

from packaging import version
from sparseml.pytorch.datasets import DatasetRegistry, VOCDetectionDataset


def _validate_voc(dataset: Dataset, size: int):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 3
    assert item[0].shape[1] == size
    assert item[0].shape[2] == size
    assert len(item[1]) == 2


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.2"),
    reason="Must install pytorch version 1.2 or greater",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False),
    reason="Skipping dataset tests",
)
def test_voc_detection():
    train_dataset = VOCDetectionDataset(train=True)
    _validate_voc(train_dataset, 300)

    val_dataset = VOCDetectionDataset(train=False)
    _validate_voc(val_dataset, 300)

    reg_dataset = DatasetRegistry.create("voc_det", train=False)
    _validate_voc(reg_dataset, 300)


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("1.2"),
    reason="Must install pytorch version 1.2 or greater",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False),
    reason="Skipping dataset tests",
)
def test_voc_segmentation():
    pass
