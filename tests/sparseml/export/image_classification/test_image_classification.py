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

from sparseml import export
from sparseml.pytorch.image_classification.utils.helpers import create_model
from sparsezoo import Model


@pytest.mark.parametrize(
    "stub, arch_key",
    [
        ("zoo:resnet_v1-50-imagenet-pruned95_quantized", "resnetv1-50"),
        ("zoo:mobilenet_v1-1.0-imagenette-base", "mobilenet_v1"),
    ],
)
class TestEndToEndExport:
    @pytest.fixture()
    def setup(self, tmp_path, stub, arch_key):
        model_path = tmp_path / "model"
        target_path = tmp_path / "target"

        self.model = Model(stub, model_path)
        self.is_model_quantized = stub.endswith("quantized")
        source_path = self.model.training.path

        # if no arch_key supplied explicitly, it can be fetched from the checkpoint
        # and thus the integration can be inferred (we can set it to None)
        integration = None if arch_key is None else "image-classification"

        kwargs = dict(num_classes=1000, image_size=224)
        if arch_key is not None:
            kwargs["arch_key"] = arch_key

        yield source_path, target_path, integration, kwargs

        shutil.rmtree(tmp_path)

    def test_export_initialized_model_no_source_path(self, setup):
        # export the image-classification model, that is being passed to the
        # `export` API directly as an object
        source_path, target_path, task, kwargs = setup
        del kwargs["num_classes"]
        kwargs["dataset_name"] = "imagenette"
        kwargs["dataset_path"] = target_path.parent / "dataset"

        model = create_model(
            checkpoint_path=os.path.join(source_path, "model.pth"),
            num_classes=10,
            **kwargs,
        )[0]

        export(
            model=model,
            target_path=target_path,
            integration="image-classification",
            validate_correctness=True,
            num_export_samples=2,
            **kwargs,
        )

        assert (target_path / "deployment" / "model.onnx").exists()
        assert not (target_path / "deployment" / "model.data").exists()

    def test_export_happy_path(self, setup):
        source_path, target_path, integration, kwargs = setup
        export(
            source_path=source_path,
            target_path=target_path,
            integration=integration,
            **kwargs,
        )
        assert (target_path / "deployment" / "model.onnx").exists()
        assert not (target_path / "deployment" / "model.data").exists()

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

    def test_export_with_external_data(self, setup):
        source_path, target_path, integration, kwargs = setup
        export(
            source_path=source_path,
            target_path=target_path,
            integration=integration,
            save_with_external_data=True,
            **kwargs,
        )
        assert (target_path / "deployment" / "model.onnx").exists()
        assert (target_path / "deployment" / "model.data").exists()

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

    def test_export_samples(self, setup):
        source_path, target_path, integration, kwargs = setup
        kwargs["dataset_name"] = "imagenette"
        kwargs["dataset_path"] = target_path.parent / "dataset"

        num_samples = 20

        export(
            source_path=source_path,
            target_path=target_path,
            integration=integration,
            num_export_samples=num_samples,
            **kwargs,
        )
        # test that the samples were properly exported
        assert (target_path / "deployment" / "model.onnx").exists()

        # download the existing samples
        self.model.sample_inputs.download()
        self.model.sample_outputs["framework"].download()
        self.model.sample_labels.download()

        # make sure that the exported data has
        # the correct structure (backward compatibility)
        self._test_exported_sample_data_structure(
            new_samples_dir=target_path / "sample-inputs",
            old_samples_dir=Path(source_path).parent / "sample-inputs",
            file_prefix="inp",
        )

        self._test_exported_sample_data_structure(
            new_samples_dir=target_path / "sample-labels",
            old_samples_dir=Path(source_path).parent / "sample-labels",
            file_prefix="lab",
        )

        self._test_exported_sample_data_structure(
            new_samples_dir=target_path / "sample-outputs",
            old_samples_dir=Path(source_path).parent / "sample-outputs",
            file_prefix="out",
        )

    def test_export_validate_correctness(self, caplog, setup):
        if self.is_model_quantized:
            pytest.skip(
                "Skipping since quantized models may not pass this test "
                "due to differences in rounding between quant ops in PyTorch and ONNX"
            )
        source_path, target_path, integration, kwargs = setup
        kwargs["dataset_name"] = "imagenette"
        kwargs["dataset_path"] = target_path.parent / "dataset"

        num_samples = 3
        export(
            source_path=source_path,
            target_path=target_path,
            integration=integration,
            num_export_samples=num_samples,
            validate_correctness=True,
            **kwargs,
        )
        assert "ERROR" not in caplog.text

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
        # outputs can have different shapes (imagenette is 10, imagenet is 1000)
        if file_prefix == "out":
            return
        if file_prefix == "lab":
            if "classes" in sample_input_old:
                # one-hot encoded labels, so we cannot compare directly
                return
        for s1, s2 in zip(sample_input_new.values(), sample_input_old.values()):
            assert s1.shape == s2.shape
