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

from torch.nn import Module

from sparseml.transformers.compression.compressors import ModelCompressor


__all__ = ["BitmaskCompressor"]


@ModelCompressor.register(name="sparse_bitmask")
class BitmaskCompressor(ModelCompressor):
    def compress(model: Module) -> Dict:
        raise NotImplementedError()

    def uncompress(model: Module, safetensors_path: str) -> Dict:
        raise NotImplementedError()
