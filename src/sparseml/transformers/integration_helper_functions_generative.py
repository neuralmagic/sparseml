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


from typing import Callable, Dict, List

from pydantic import Field

from sparseml.transformers.integration_helper_functions import Transformers
from sparseml.transformers.utils.helpers import (
    MANDATORY_DEPLOYMENT_FILES,
    NLG_TOKENIZER_FILES,
)
from sparseml.transformers.utils.optimizations import apply_kv_cache_injection
from src.sparseml.integration_helper_functions import (
    IntegrationHelperFunctions,
    Integrations,
)


generative_transformers_graph_optimizations = {
    "kv_cache_injection": apply_kv_cache_injection
}


@IntegrationHelperFunctions.register(name=Integrations.transformers_generative.value)
class GenerativeTransformers(Transformers):
    graph_optimizations: Dict[str, Callable] = Field(
        default=generative_transformers_graph_optimizations
    )
    deployment_directory_files_mandatory: List[str] = Field(
        default=list(MANDATORY_DEPLOYMENT_FILES.union(NLG_TOKENIZER_FILES))
    )
