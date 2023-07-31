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

import onnx
import pytest

from sparseml.exporters.kv_cache_injector import KeyValueCacheInjector
from sparsezoo import Model


@pytest.mark.parametrize(
    "stub, expected_input_names",
    [
        # codegen
        (
            "zoo:nlg/text_generation/codegen_multi-350m/pytorch/huggingface/bigquery_thepile/pruned50_quant-none",  # noqa: E501
            ["input_ids", "attention_mask", "causal_mask", "positions"],
        ),
        # opt
        (
            "zoo:nlg/text_generation/opt-1.3b/pytorch/huggingface/opt_pretrain/pruned50_quantW8A8-none",  # noqa: E501
            ["input_ids", "attention_mask", "causal_mask", "positions"],
        ),
    ],
    scope="class",
)
class TestKeyValueCacheInjector:
    @pytest.fixture()
    def setup(self, stub, expected_input_names):
        sparsezoo_model = Model(stub)

        # load the model with no cache injected and inject the cache
        model_no_cache = onnx.load(
            sparsezoo_model.training.get_file("model.onnx").path,
            load_external_data=False,
        )
        model_cache_injected = KeyValueCacheInjector(
            sparsezoo_model.deployment.path
        ).apply(model_no_cache)

        yield model_cache_injected, expected_input_names, sparsezoo_model

    def test_models_identical(self, setup):
        model_cache_injected, _, sparsezoo_model = setup
        # the model with cache injected should be the same as the model in deployment
        model_baseline = onnx.load(
            sparsezoo_model.deployment.get_file("model.onnx").path,
            load_external_data=False,
        )

    def test_inputs_names(self, setup):
        model_cache_injected, expected_input_names, _ = setup
        assert set(input.name for input in model_cache_injected.graph.input) == set(
            expected_input_names
        )

    def test_output_names(self, setup):
        model_cache_injected, _, _ = setup
        assert all(
            input.name.startswith("past_key_values") or input.name == "logits"
            for input in model_cache_injected.graph.output
        )
