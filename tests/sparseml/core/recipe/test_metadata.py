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


import pytest

from sparseml.core.recipe.metadata import ModelMetaData, RecipeMetaData


class TestRecipeMetaData:
    @pytest.mark.parametrize(
        "self_metadata",
        [
            dict(domain="cv", task="classification"),
            dict(),
        ],
    )
    @pytest.mark.parametrize(
        "other_metadata",
        [
            dict(domain="domain", task="segmentation", requirements=["torch>=1.6.0"]),
            dict(
                domain="cv",
                task="task",
                target_model=ModelMetaData(layer_prefix="something"),
            ),
        ],
    )
    def test_update_missing_metadata(self, self_metadata, other_metadata):

        metadata_a = RecipeMetaData(**self_metadata)
        metadata_b = RecipeMetaData(**other_metadata)

        metadata_a.update_missing_metadata(metadata_b)

        all_keys = set(self_metadata.keys()).union(other_metadata.keys())

        # keys should not be overwritten
        # if they already exist
        for key in all_keys:
            if key in self_metadata:
                assert getattr(metadata_a, key) == self_metadata[key]
            elif key in other_metadata:
                assert getattr(metadata_a, key) == other_metadata[key]
