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
import tarfile
from enum import Enum

import pytest

from sparseml.export.export_data import create_data_samples, export_data_sample


@pytest.fixture()
def dummy_names():
    class LabelNames(Enum):
        basename = "sample-dummies"
        filename = "dummy"

    return LabelNames


@pytest.fixture()
def dummy_samples():
    import torch

    num_samples = 5
    batch_size = 3
    samples = [torch.randn(batch_size, 3, 224, 224) for _ in range(num_samples)]

    return samples


@pytest.mark.parametrize(
    "as_tar",
    [True, False],
)
def test_export_data_sample(tmp_path, as_tar, dummy_names, dummy_samples):
    export_data_sample(
        samples=dummy_samples, names=dummy_names, target_path=tmp_path, as_tar=as_tar
    )

    dir_name = dummy_names.basename.value
    dir_name_tar = dummy_names.basename.value + ".tar.gz"

    if as_tar:
        with tarfile.open(os.path.join(tmp_path, dir_name_tar)) as tar:
            tar.extractall(path=tmp_path)

    assert (
        set(os.listdir(tmp_path)) == {dir_name}
        if not as_tar
        else {dir_name, dir_name_tar}
    )
    assert set(os.listdir(os.path.join(tmp_path, "sample-dummies"))) == {
        f"dummy-000{i}.npz" for i in range(len(dummy_samples))
    }


@pytest.mark.parametrize(
    "model",
    [True, False],
)
@pytest.mark.parametrize("num_samples", [0, 1, 5])
@pytest.mark.parametrize("scenario", ["transformers", "image_classification"])
def test_create_data_samples(num_samples, model, scenario):
    pytest.importorskip("torch", reason="test requires pytorch")
    if scenario == "transformers":
        # TODO: leaving it here to add appropriate test cases
        # before landing to main
        assert False

    import torch
    from torch.utils.data import DataLoader, Dataset

    model = torch.nn.Sequential(torch.nn.Identity()) if model else None

    class DummyDataset(Dataset):
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
    labels = torch.randint(
        0,
        10,
        (
            100,
            50,
        ),
    )

    custom_dataset = DummyDataset(inputs, labels)

    data_loader = DataLoader(custom_dataset, batch_size=1)

    inputs, outputs, labels = create_data_samples(
        data_loader=data_loader, num_samples=num_samples, model=model
    )

    assert all(tuple(input.shape) == (1, 3, 224, 224) for input in inputs)
    assert all(tuple(label.shape) == (1, 50) for label in labels)
    assert len(inputs) == num_samples == len(labels)

    if model is not None:
        assert len(outputs) == num_samples
        assert all(tuple(output.shape) == (1, 3, 224, 224) for output in outputs)
