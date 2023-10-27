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

from sparseml.transformers.finetune.data import TextGenerationDataset


@TextGenerationDataset.register(name="c4")
class C4Dataset(TextGenerationDataset):
    def __init__(self, data_args, tokenizer):
        raw_kwargs = {"data_files": "en/c4-train.00000-of-01024.json.gz"}
        super().__init__(text_column="text", data_args=data_args, tokenizer=tokenizer, raw_kwargs=raw_kwargs)
