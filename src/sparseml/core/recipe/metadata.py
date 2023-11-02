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

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


__all__ = [
    "NMVersions",
    "DatasetMetaData",
    "ParamMetaData",
    "LayerMetaData",
    "ModelMetaData",
    "RecipeMetaData",
]


class NMVersions(BaseModel):
    sparsezoo_version: str = None
    sparsezoo_hash: str = None
    sparseml_version: str = None
    sparseml_hash: str = None
    sparsify_version: str = None
    sparsify_hash: str = None


class DatasetMetaData(BaseModel):
    name: str = None
    version: str = None
    hash: str = None
    shape: List[int] = Field(default_factory=list)
    num_classes: int = None
    num_train_samples: int = None
    num_val_samples: int = None
    num_test_samples: int = None


class ParamMetaData(BaseModel):
    name: str = None
    shape: List[int] = None
    weight_hash: str = None


class LayerMetaData(BaseModel):
    name: str = None
    type: str = None
    index: int = None
    attributes: Dict[str, Any] = None
    input_shapes: List[List[int]] = None
    output_shapes: List[List[int]] = None
    params: Dict[str, ParamMetaData] = None


class ModelMetaData(BaseModel):
    architecture: str = None
    sub_architecture: str = None
    input_shapes: List[List[int]] = None
    output_shapes: List[List[int]] = None
    layers: List[LayerMetaData] = Field(default_factory=list)
    layer_prefix: Optional[str] = None


class RecipeMetaData(BaseModel):
    domain: str = None
    task: str = None
    versions: NMVersions = Field(default_factory=NMVersions)
    requirements: List[str] = None
    tags: List[str] = None
    target_dataset: DatasetMetaData = None
    target_model: ModelMetaData = None

    def update_missing_metadata(self, other: "RecipeMetaData"):
        """
        Update recipe metadata with missing values from another
        recipe metadata instance

        :param other: the recipe metadata to update with
        """
        self.domain = self.domain or other.domain
        self.task = self.task or other.task
        self.versions = self.versions or other.versions
        self.requirements = self.requirements or other.requirements
        self.tags = self.tags or other.tags
        self.target_dataset = self.target_dataset or other.target_dataset
        self.target_model = self.target_model or other.target_model
