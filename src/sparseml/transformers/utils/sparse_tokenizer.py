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

from transformers import AutoTokenizer

from sparsezoo import Model


__all__ = ["SparseAutoTokenizer"]


class SparseAutoTokenizer(AutoTokenizer):
    """
    SparseML wrapper for the AutoTokenizer class
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        A wrapper around the AutoTokenizer.from_pretrained method that
        enables the loading of tokenizer from SparseZoo stubs

        :param pretrained_model_name_or_path: the name of or path to the model to load
        :return tokenizer: the loaded tokenizer from pretrained
        """
        if pretrained_model_name_or_path.startswith("zoo:"):
            # if SparseZoo stub is passed, fetch the deployment
            # path of the SparseZoo model and replace
            # pretrained_model_name_or_path with the deployment path
            pretrained_model_name_or_path = Model(
                pretrained_model_name_or_path
            ).deployment.path

        return super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
