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
from tests.sparseml.export.utils import get_dummy_dataset


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
def test_create_data_samples_transformers(num_samples, model):
    pytest.importorskip("torch", reason="test requires pytorch")
    import torch
    from torch.utils.data import DataLoader

    class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.empty(0))
            self.device = self.dummy_param.device

        def forward(self, input_ids, attention_mask):
            return dict(input_ids=input_ids, attention_mask=attention_mask)

    model = Identity().to("cpu") if model else None

    data_loader = DataLoader(get_dummy_dataset("transformers"), batch_size=1)

    inputs, outputs, labels = create_data_samples(
        data_loader=data_loader, num_samples=num_samples, model=model
    )
    target_input = next(iter(data_loader))
    target_output = target_input

    assert len(inputs) == num_samples
    for input in inputs:
        for key, value in input.items():
            assert torch.equal(value, target_input[key])
    assert labels == []
    if model is not None:
        assert len(outputs) == num_samples
        for output in outputs:
            for key, value in output.items():
                assert torch.equal(value, target_output[key][0])


@pytest.mark.parametrize(
    "model",
    [True, False],
)
@pytest.mark.parametrize("num_samples", [0, 1, 5])
def test_create_data_samples_image_classification(num_samples, model):
    pytest.importorskip("torch", reason="test requires pytorch")

    import torch
    from torch.utils.data import DataLoader

    model = torch.nn.Sequential(torch.nn.Identity()) if model else None
    data_loader = DataLoader(get_dummy_dataset("image-classification"), batch_size=1)

    inputs, outputs, labels = create_data_samples(
        data_loader=data_loader, num_samples=num_samples, model=model
    )
    target_input, target_label = next(iter(data_loader))
    target_output = target_input
    assert all(input.shape == target_input.shape for input in inputs)
    assert all(label.shape == target_label.shape for label in labels)
    assert len(inputs) == num_samples == len(labels)

    if model is not None:
        assert len(outputs) == num_samples
        assert all(output.shape == target_output.shape for output in outputs)
