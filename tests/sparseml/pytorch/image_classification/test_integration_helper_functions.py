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
    import sparseml.pytorch.image_classification.integration_helper_functions  # noqa F401

    image_classification = IntegrationHelperFunctions.load_from_registry(
        Integrations.image_classification.value
    )
    assert image_classification.create_model
    assert image_classification.create_dummy_input
    assert image_classification.export
    assert image_classification.graph_optimizations is None
    assert image_classification.create_data_samples
