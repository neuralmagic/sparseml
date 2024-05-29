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

import numpy as np
import onnx
import pytest
import torch

from huggingface_hub import snapshot_download
from sparseml import export
from sparseml.transformers import SparseAutoModelForCausalLM, SparseAutoTokenizer


@pytest.mark.parametrize(
    "stub, task",
    [("roneneldan/TinyStories-1M", "text-generation")],
)
class TestEndToEndExport:
    @pytest.fixture()
    def setup(self, tmp_path, stub, task):
        model_path = tmp_path / "model"
        target_path = tmp_path / "target"

        source_path = snapshot_download(stub, local_dir=model_path)

        yield source_path, target_path, task

        shutil.rmtree(tmp_path)

    def test_export_initialized_model_no_source_path(self, tmp_path, stub, task):
        # export the transformer model, that is being passed to the
        # `export` API directly as an object
        target_path = tmp_path / "target"
        export(
            model=SparseAutoModelForCausalLM.from_pretrained(stub),
            tokenizer=SparseAutoTokenizer.from_pretrained(stub),
            target_path=target_path,
            sequence_length=384,
            task=task,
            validate_correctness=True,
            num_export_samples=2,
            **dict(
                data_args=dict(dataset="ultrachat-200k", dataset_config_name="default")
            ),
        )
        assert (target_path / "deployment" / "model.onnx").exists()
        assert not (target_path / "deployment" / "model.data").exists()
        # check if kv cache injection has been applied
        onnx_model = onnx.load(
            str(target_path / "deployment" / "model.onnx"), load_external_data=False
        )
        assert any(
            inp.name == "past_key_values.0.key" for inp in onnx_model.graph.input
        )

        shutil.rmtree(tmp_path)

    def test_export_happy_path(self, setup):
        source_path, target_path, task = setup
        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
        )
        assert (target_path / "deployment" / "model.onnx").exists()
        assert (target_path / "deployment" / "model-orig.onnx").exists()
        assert not (target_path / "deployment" / "model.data").exists()
        # check if kv cache injection has been applied
        onnx_model = onnx.load(
            str(target_path / "deployment" / "model.onnx"), load_external_data=False
        )
        assert any(
            inp.name == "past_key_values.0.key" for inp in onnx_model.graph.input
        )

    def text_export_without_optimizations(self, setup):
        source_path, target_path, task = setup
        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
            graph_optimizations="none",
            # set validate correctness to True to ensure that
            # samples are correctly generated and consumed
            # by the model
            validate_correctness=True,
            **dict(
                data_args=dict(
                    dataset="wikitext", dataset_config_name="wikitext-2-raw-v1"
                )
            ),
        )
        assert (target_path / "deployment" / "model.onnx").exists()
        assert not (target_path / "deployment" / "model-orig.onnx").exists()
        # check if that kv cache injection has not been applied
        onnx_model = onnx.load(
            str(target_path / "deployment" / "model.onnx"), load_external_data=False
        )
        assert not all(
            inp.name == "past_key_values.0.key" for inp in onnx_model.graph.input
        )

    def test_export_with_external_data(self, setup):
        source_path, target_path, task = setup
        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
            save_with_external_data=True,
        )
        assert (target_path / "deployment" / "model.onnx").exists()
        assert (target_path / "deployment" / "model.data").exists()

    def test_export_with_recipe(self, setup):
        source_path, target_path, task = setup

        recipe = """test_stage:
          quant_modifiers:
            LegacyQuantizationModifier:
              post_oneshot_calibration: False
              scheme_overrides:
                Embedding:
                  input_activations: null"""

        with open(os.path.join(source_path, "recipe.yaml"), "w") as f:
            f.write(recipe)

        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
        )
        assert (target_path / "deployment" / "model.onnx").exists()

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

    def test_export_samples(self, setup):
        source_path, target_path, task = setup

        num_samples = 4

        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
            num_export_samples=num_samples,
            **dict(
                data_args=dict(
                    dataset="wikitext", dataset_config_name="wikitext-2-raw-v1"
                )
            ),
        )

        assert (target_path / "deployment" / "model.onnx").exists()
        assert (
            len(os.listdir(os.path.join(target_path, "sample-inputs"))) == num_samples
        )
        assert (
            len(os.listdir(os.path.join(target_path, "sample-outputs"))) == num_samples
        )
        sample_input_new = np.load(
            os.path.join(target_path, "sample-outputs", "out-0000.npz")
        )
        assert ["logits"] == list(sample_input_new.keys())

    def test_export_validate_correctness(self, caplog, setup):
        source_path, target_path, task = setup

        num_samples = 3

        export(
            source_path=source_path,
            target_path=target_path,
            task=task,
            num_export_samples=num_samples,
            validate_correctness=True,
            **dict(
                data_args=dict(
                    dataset="wikitext", dataset_config_name="wikitext-2-raw-v1"
                )
            ),
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
            **dict(
                data_args=dict(
                    dataset="wikitext", dataset_config_name="wikitext-2-raw-v1"
                )
            ),
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
            **dict(
                data_args=dict(
                    dataset="wikitext", dataset_config_name="wikitext-2-raw-v1"
                )
            ),
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
