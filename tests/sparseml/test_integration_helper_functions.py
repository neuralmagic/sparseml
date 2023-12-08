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

import pytest

from sparseml.integration_helper_functions import create_sample_inputs_outputs


@pytest.mark.parametrize("num_samples", [0, 1, 5])
def test_create_sample_inputs_outputs(num_samples):
    pytest.importorskip("torch", reason="test requires pytorch")
    import torch
    from torch.utils.data import DataLoader, Dataset

    class DummmyDataset(Dataset):
        def __init__(self, inputs, outputs):
            self.data = inputs
            self.target = outputs

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            data_sample = self.data[index]
            target_sample = self.target[index]

            return data_sample, target_sample

    inputs = torch.randn((100, 3, 224, 224))
    outputs = torch.randint(
        0,
        10,
        (
            100,
            50,
        ),
    )

    custom_dataset = DummmyDataset(inputs, outputs)

    data_loader = DataLoader(custom_dataset, batch_size=1)

    inputs, outputs = create_sample_inputs_outputs(data_loader, num_samples)

    assert all(tuple(input.shape) == (1, 3, 224, 224) for input in inputs)
    assert all(tuple(output.shape) == (1, 50) for output in outputs)

    assert len(inputs) == num_samples == len(outputs)
