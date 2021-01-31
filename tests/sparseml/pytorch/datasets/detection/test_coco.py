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


try:
    import pycocotools
except Exception:
    pycocotools = None

from sparseml.pytorch.datasets import CocoDetectionDataset, DatasetRegistry


def _validate_coco(dataset: Dataset, size: int):
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert item[0].shape[0] == 3
    assert item[0].shape[1] == size
    assert item[0].shape[2] == size
    assert len(item[1]) == 2


@pytest.mark.skipif(pycocotools is None, reason="Pycocotools not installed")
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_DATASET_TESTS", False),
    reason="Skipping dataset tests",
)
def test_coco_detection():
    # 18 GB download
    train_dataset = CocoDetectionDataset(train=True)
    _validate_coco(train_dataset, 300)

    val_dataset = CocoDetectionDataset(train=False)
    _validate_coco(val_dataset, 300)

    reg_dataset = DatasetRegistry.create("coco", train=False)
    _validate_coco(reg_dataset, 300)
