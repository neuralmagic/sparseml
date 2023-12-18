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


import glob
import os
import shutil

import numpy as np
import pytest
import torch

from sparseml.export.export import export
from sparsezoo import Model


@pytest.mark.parametrize(
    "stub, task",
    [
        ("zoo:obert-medium-squad_wikipedia_bookcorpus-pruned95_quantized", "qa"),
    ],
)
class TestEndToEndExport:
    @pytest.fixture()
    def setup(self, tmp_path, stub, task):
        model_path = tmp_path / "model"
        target_path = tmp_path / "target"

        source_path = Model(stub, model_path).training.path

        yield source_path, target_path, task

        shutil.rmtree(tmp_path)

    def test_export_happy_path(self, setup):
        source_path, target_path, task = setup
        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
        )
        assert (target_path / "deployment" / "model.onnx").exists()

    def test_export_samples(self, setup):
        source_path, target_path, task = setup

        num_samples = 4

        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
            num_export_samples=num_samples,
            **dict(data_args=dict(dataset_name="squad")),
        )
        assert (target_path / "deployment" / "model.onnx").exists()
        assert (
            len(os.listdir(os.path.join(target_path, "sample-inputs"))) == num_samples
        )
        assert (
            len(os.listdir(os.path.join(target_path, "sample-outputs"))) == num_samples
        )
        assert np.load(
            glob.glob(os.path.join(target_path, "sample-inputs/*"))[0],
            allow_pickle=True,
        )["arr_0"]

    def test_export_with_sample_data(self, setup):
        source_path, target_path, task = setup

        sequence_length = 32
        sample_data = dict(
            input_ids=torch.ones((10, sequence_length), dtype=torch.long),
            attention_mask=torch.ones((10, sequence_length), dtype=torch.long),
        )
        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
            sample_data=sample_data,
        )
        assert (target_path / "deployment" / "model.onnx").exists()

    @pytest.mark.skipif(reason="skipping since not implemented")
    def test_export_multiple_files(self, setup):
        source_path, target_path, task = setup
        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
            single_graph_file=False,
        )

    @pytest.mark.skipif(
        reason="skipping since this functionality needs some more attention"
    )
    def test_export_validate_correctness(self, setup):
        source_path, target_path, task = setup

        num_samples = 4

        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
            num_export_samples=num_samples,
            validate_correctness=True,
            **dict(data_args=dict(dataset_name="squad")),
        )
