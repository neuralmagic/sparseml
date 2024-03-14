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

from typing import Dict

from torch import Tensor

from sparseml.transformers.compression.config import CompressionConfig
from sparsezoo.utils.registry import RegistryMixin


__all__ = ["ModelCompressor"]


class ModelCompressor(RegistryMixin):
    """
    Base class representing a model compression algorithm.

    :param config: config specifying compression parameters
    """

    def __init__(self, config: CompressionConfig):
        self.config = config

    def compress(self, model_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compresses a dense state dict

        :param model_state: state dict of uncompressed model
        :return: compressed state dict
        """
        raise NotImplementedError()

    def decompress(self, model_path: str) -> Dict[str, Tensor]:
        """
        Uncompresses a compressed state dict back to dense

        :param model_state: state dict of uncompressed model
        :return: compressed state dict
        """
        raise NotImplementedError()
