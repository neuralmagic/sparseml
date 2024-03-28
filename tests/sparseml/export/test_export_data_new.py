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
import shutil
import tarfile
import unittest
from enum import Enum
from pathlib import Path

import pytest

from parameterized import parameterized
from sparseml.export.export_data import create_data_samples, export_data_sample
from tests.sparseml.export.utils import get_dummy_dataset
from tests.testing_utils import requires_torch


# NOTE: These tests are equivalent to the tests in test_export_data, updated to use
# the new framework


@requires_torch
@pytest.mark.unit
class ExportDataTransformersUnitTest(unittest.TestCase):
    def setUp(self):
        import torch
        from torch.utils.data import DataLoader

        class Identity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = torch.nn.Parameter(torch.empty(0))
                self.device = self.dummy_param.device

            def forward(self, input_ids, attention_mask):
                return dict(input_ids=input_ids, attention_mask=attention_mask)

        self.identity_model = Identity()
        self.data_loader = DataLoader(get_dummy_dataset("transformers"), batch_size=1)

    @parameterized.expand(
        [[0, True], [0, False], [1, True], [1, False], [5, True], [5, False]]
    )
    def test_create_data_samples(self, num_samples, model):
        import torch

        model = self.identity_model.to("cpu") if model else None

        inputs, outputs, labels = create_data_samples(
            data_loader=self.data_loader, num_samples=num_samples, model=model
        )
        target_input = next(iter(self.data_loader))
        target_output = target_input

        self.assertEqual(len(inputs), num_samples)
        for input in inputs:
            for key, value in input.items():
                assert torch.equal(value.unsqueeze(0), target_input[key])
        assert labels == []
        if model is not None:
            assert len(outputs) == num_samples
            for output in outputs:
                for key, value in output.items():
                    assert torch.equal(value, target_output[key][0])

    def tearDown(self):
        pass


@requires_torch
@pytest.mark.unit
class ExportGenericDataUnitTest(unittest.TestCase):
    def setUp(self):
        import torch

        class LabelNames(Enum):
            basename = "sample-dummies"
            filename = "dummy"

        num_samples = 5
        batch_size = 3
        self.samples = [
            torch.randn(batch_size, 3, 224, 224) for _ in range(num_samples)
        ]
        self.names = LabelNames
        self.tmp_path = Path("tmp")
        self.tmp_path.mkdir(exist_ok=True)

    @parameterized.expand([[True], [False]])
    def test_export_data_sample(self, as_tar):
        export_data_sample(
            samples=self.samples,
            names=self.names,
            target_path=self.tmp_path,
            as_tar=as_tar,
        )

        dir_name = self.names.basename.value
        dir_name_tar = self.names.basename.value + ".tar.gz"

        if as_tar:
            with tarfile.open(os.path.join(self.tmp_path, dir_name_tar)) as tar:
                tar.extractall(path=self.tmp_path)

        assert (
            set(os.listdir(self.tmp_path)) == {dir_name}
            if not as_tar
            else {dir_name, dir_name_tar}
        )
        assert set(os.listdir(os.path.join(self.tmp_path, "sample-dummies"))) == {
            f"dummy-000{i}.npz" for i in range(len(self.samples))
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_path)


# NOTE: Dummy smoke test


@pytest.mark.smoke
@requires_torch
class ExportDataDummySmokeTest(unittest.TestCase):
    def setUp(self):
        import torch

        self.samples = [torch.randn(1, 3, 224, 224) for _ in range(2)]

        class LabelNames(Enum):
            basename = "sample-dummies"
            filename = "dummy"

        self.names = LabelNames

    @parameterized.expand([["some_path"], [Path("some_path")]])
    def test_export_runs(self, target_path):
        Path(target_path).mkdir(exist_ok=True)
        export_data_sample(
            samples=self.samples, names=self.names, target_path=target_path
        )
        shutil.rmtree(target_path)
