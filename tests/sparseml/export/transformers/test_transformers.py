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
from pathlib import Path

import numpy as np
import pytest
import torch

from sparseml.export.export import export
from sparsezoo import Model


@pytest.mark.parametrize(
    "stub, task",
    [
        ("zoo:obert-medium-squad_wikipedia_bookcorpus-pruned95_quantized", "qa"),
        ("zoo:roberta-large-squad_v2_wikipedia_bookcorpus-base", "qa"),
    ],
)
class TestEndToEndExport:
    @pytest.fixture()
    def setup(self, tmp_path, stub, task):
        model_path = tmp_path / "model"
        target_path = tmp_path / "target"
        self.model = Model(stub, model_path)
        self.is_model_quantized = stub.endswith("quantized")
        source_path = self.model.training.path

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

        num_samples = 20

        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
            num_export_samples=num_samples,
            **dict(data_args=dict(dataset_name="squad")),
        )
        assert (target_path / "deployment" / "model.onnx").exists()

        # download the existing samples
        self.model.sample_inputs.download()
        self.model.sample_outputs["framework"].download()

        # make sure that the exported data has
        # the correct structure (backward compatibility)
        self._test_exported_sample_data_structure(
            new_samples_dir=target_path / "sample-inputs",
            old_samples_dir=Path(source_path).parent / "sample-inputs",
            file_prefix="inp",
        )

        self._test_exported_sample_data_structure(
            new_samples_dir=target_path / "sample-outputs",
            old_samples_dir=Path(source_path).parent / "sample-outputs",
            file_prefix="out",
        )

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

    def test_export_validate_correctness(self, caplog, setup):
        if self.is_model_quantized:
            pytest.skip(
                "Skipping since quantized models may not pass this test"
                "due to differences in rounding between quant ops in PyTorch and ONNX"
            )
        source_path, target_path, task = setup

        num_samples = 3

        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
            num_export_samples=num_samples,
            validate_correctness=True,
            **dict(data_args=dict(dataset_name="squad")),
        )

        assert "ERROR" not in caplog.text

    def test_export_multiple_times(self, caplog, setup):
        # make sure that when we export multiple times,
        # the user gets verbose warning about the files
        # already existing and being overwritten
        source_path, target_path, task = setup

        num_samples = 3

        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
            num_export_samples=num_samples,
            **dict(data_args=dict(dataset_name="squad")),
        )
        warnings_after_first_export = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]
        caplog.clear()

        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
            num_export_samples=num_samples,
            **dict(data_args=dict(dataset_name="squad")),
        )
        warnings_after_second_export = [
            record.message for record in caplog.records if record.levelname == "WARNING"
        ]

        new_warnings = set(warnings_after_second_export) - set(
            warnings_after_first_export
        )

        # make sure that all the unique warnings that happen only after the
        # repeated export are about the files already existing
        for warning in new_warnings:
            assert "already exist" in warning
            assert "Overwriting" in warning

    @staticmethod
    def _test_exported_sample_data_structure(
        new_samples_dir, old_samples_dir, file_prefix
    ):
        assert new_samples_dir.exists()
        assert set(os.listdir(new_samples_dir)) == set(os.listdir(old_samples_dir))

        # read the first sample from the newly
        # generated samples and the downloaded samples
        sample_input_new = np.load(
            os.path.join(new_samples_dir, f"{file_prefix}-0000.npz")
        )
        sample_input_old = np.load(
            os.path.join(old_samples_dir, f"{file_prefix}-0000.npz")
        )

        for s1, s2 in zip(sample_input_new.values(), sample_input_old.values()):
            assert s1.shape == s2.shape
