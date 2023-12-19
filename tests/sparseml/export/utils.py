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

from typing import Dict, List

import torch
from torch.utils.data import Dataset


class DummyDatasetTransformers(Dataset):
    def __init__(self, inputs: List[Dict]):
        self.data = inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_sample = self.data[index]
        return data_sample


class DummyDatasetImageClassification(Dataset):
    def __init__(self, inputs: torch.Tensor, labels: torch.Tensor):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        data_sample = self.inputs[index]
        label_sample = self.labels[index]
        return data_sample, label_sample


def get_dummy_dataset(integration):
    if integration == "image-classification":
        inputs = torch.randn((100, 3, 224, 224))
        labels = torch.randint(
            0,
            10,
            (
                100,
                50,
            ),
        )
        return DummyDatasetImageClassification(inputs, labels)
    elif integration == "transformers":
        input = dict(
            input_ids=torch.ones((10, 100), dtype=torch.long),
            attention_mask=torch.ones((10, 100), dtype=torch.long),
        )
        return DummyDatasetTransformers([input for _ in range(100)])
    else:
        raise NotImplementedError(
            "Getting dummy dataset for " f"integration {integration} not implemented"
        )
