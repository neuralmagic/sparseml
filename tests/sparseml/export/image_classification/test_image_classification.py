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
    "stub, arch_key", [("zoo:resnet_v1-50-imagenet-pruned95_quantized", "resnetv1-50")]
)
class TestEndToEndExport:
    @pytest.fixture()
    def setup(self, tmp_path, stub, arch_key):
        model_path = tmp_path / "model"
        target_path = tmp_path / "target"

        source_path = Model(stub, model_path).training.path

        # if no arch_key supplied explicitly, it can be fetched from the checkpoint
        # and thus the integration can be inferred (we can set it to None)
        integration = None if arch_key is None else "image-classification"

        kwargs = dict(num_classes=1000, image_size=224)
        if arch_key is not None:
            kwargs["arch_key"] = arch_key

        yield source_path, target_path, integration, kwargs

        shutil.rmtree(tmp_path)

    def test_export_happy_path(self, setup):
        source_path, target_path, integration, kwargs = setup
        export(
            source_path=source_path,
            target_path=target_path,
            integration=integration,
            **kwargs,
        )
        assert (target_path / "deployment" / "model.onnx").exists()

    def test_export_custom_onnx_model_name(self, setup):
        source_path, target_path, integration, kwargs = setup
        export(
            source_path=source_path,
            target_path=target_path,
            integration=integration,
            onnx_model_name="custom_model_name.onnx",
            **kwargs,
        )
        assert (target_path / "deployment" / "custom_model_name.onnx").exists()

    def test_export_custom_deployment_name(self, setup):
        source_path, target_path, integration, kwargs = setup
        export(
            source_path=source_path,
            target_path=target_path,
            integration=integration,
            deployment_directory_name="custom_deployment_name",
            **kwargs,
        )
        assert (target_path / "custom_deployment_name" / "model.onnx").exists()

    def test_export_deployment_target_onnx(self, setup):
        source_path, target_path, integration, kwargs = setup
        export(
            source_path=source_path,
            target_path=target_path,
            integration=integration,
            deployment_target="onnx",
            **kwargs,
        )
        assert (target_path / "deployment" / "model.onnx").exists()

    def test_export_with_sample_data(self, setup):
        source_path, target_path, integration, kwargs = setup
        export(
            source_path=source_path,
            target_path=target_path,
            integration=integration,
            sample_data=torch.randn(1, 3, 224, 224),
            **kwargs,
        )
        assert (target_path / "deployment" / "model.onnx").exists()

    @pytest.mark.skipif(reason="skipping since not implemented")
    def test_export_multiple_files(self, setup):
        source_path, target_path, integration, kwargs = setup
        export(
            source_path=source_path,
            target_path=target_path,
            integration=integration,
            single_graph_file=False,
            **kwargs,
        )

    def test_export_samples(self, setup):
        source_path, target_path, integration, kwargs = setup
        del kwargs["num_classes"]
        kwargs["dataset_name"] = "imagenette"
        kwargs["dataset_path"] = target_path.parent / "dataset"

        num_samples = 10
        batch_size = 2

        export(
            source_path=source_path,
            target_path=target_path,
            integration=integration,
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

    @pytest.mark.skipif(
        reason="skipping since this functionality needs some more attention"
    )
    def test_export_validate_correctness(self, setup):
        source_path, target_path, integration, kwargs = setup
        del kwargs["num_classes"]
        kwargs["dataset_name"] = "imagenette"
        kwargs["dataset_path"] = target_path.parent / "dataset"

        num_samples = 10
        batch_size = 2

        export(
            source_path=source_path,
            target_path=target_path,
            integration=integration,
            num_export_samples=num_samples,
            batch_size=batch_size,
            validate_correctness=True,
            **kwargs,
        )
