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
from src.sparseml.integration_helper_functions import (
    IntegrationHelperFunctions,
    Integrations,
)


def test_integration_helper_functions():
    # import needed to register the object on the fly
    import sparseml.transformers.integration_helper_functions  # noqa F401

    transformers = IntegrationHelperFunctions.load_from_registry(
        Integrations.transformers.value
    )
    assert transformers.create_model
    assert transformers.create_dummy_input
    assert transformers.export
    assert transformers.graph_optimizations is None
    assert transformers.create_data_samples
    assert set(transformers.deployment_directory_files_mandatory) == {
        "model.onnx",
        "tokenizer_config.json",
        "config.json",
    }
    assert set(transformers.deployment_directory_files_optional) == {
        "tokenizer.json",
        "tokenizer.model",
    }


def test_integration_helper_function_text_generation():
    # import needed to register the object on the fly
    import sparseml.transformers.integration_helper_functions  # noqa F401

    transformers = IntegrationHelperFunctions.load_from_registry(
        Integrations.transformers.value, task="text-generation"
    )

    assert transformers.create_model
    assert transformers.create_dummy_input
    assert transformers.export
    assert set(transformers.graph_optimizations.keys()) == {"kv_cache_injection"}
    assert transformers.create_data_samples
    assert set(transformers.deployment_directory_files_mandatory) == {
        "model.onnx",
        "tokenizer_config.json",
        "config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    }
    assert set(transformers.deployment_directory_files_optional) == {
        "tokenizer.json",
        "tokenizer.model",
    }
