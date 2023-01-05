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

from abc import abstractmethod
from typing import Any, List

from sparseml.exporters.transforms import BaseTransform


class BaseExporter(BaseTransform):
    def __init__(self, transforms: List[BaseTransform]) -> None:
        super().__init__()
        self.transforms = transforms

    def transform(self, model: Any) -> Any:
        for transform in self.transforms:
            model = transform.apply(model)
        return model

    @abstractmethod
    def export(self, pre_transforms_model: Any, file_path: str):
        """
        Applies all the transforms to the model and then saves the model to `path`.

        :param pre_transforms_model: The model to export, passed into the list of
            transforms on this exporter.
        :param file_path: The path to save the post transformed model to
        """
        raise NotImplementedError
