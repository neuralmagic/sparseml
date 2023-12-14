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

from sparseml.export.export import export
from sparsezoo import Model


@pytest.mark.parametrize(
    "stub, task",
    [("zoo:obert-medium-squad_wikipedia_bookcorpus-pruned95_quantized", "qa")],
)
class TestEndToEndExport:
    @pytest.fixture()
    def setup(self, tmp_path, stub, task):
        model_path = tmp_path / "model"
        target_path = tmp_path / "target"

        source_path = Model(stub, model_path).training.path
        kwargs = dict(task=task)
        yield source_path, target_path, kwargs

        shutil.rmtree(tmp_path)

    def test_export_happy_path(self, setup):
        source_path, target_path, kwargs = setup
        export(
            source_path=source_path,
            target_path=target_path,
            **kwargs,
        )
        assert (target_path / "deployment" / "model.onnx").exists()

    def test_export_samples(self, setup):
        source_path, target_path, kwargs = setup
        kwargs["data_args"] = dict(dataset_name="squad")

        num_samples = 4
        batch_size = 2

        export(
            source_path=source_path,
            target_path=target_path,
            num_export_samples=num_samples,
            batch_size=batch_size,
            **kwargs,
        )
        assert (target_path / "deployment" / "model.onnx").exists()
        assert (
            len(os.listdir(os.path.join(target_path, "sample-labels"))) == num_samples
        )
        assert (
            len(os.listdir(os.path.join(target_path, "sample-inputs"))) == num_samples
        )
        assert (
            len(os.listdir(os.path.join(target_path, "sample-outputs"))) == num_samples
        )
        # # open the sample-inputs file and check the batch size
        sample_input = np.load(
            glob.glob(os.path.join(target_path, "sample-inputs/*"))[0]
        )["arr_0"]
        assert sample_input.shape[0] == batch_size
