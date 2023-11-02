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

from torch.nn import Module

from sparseml.transformers.data.base_llm import TransformersDataset


@TransformersDataset.register(name="ptb")
class Ptb(TransformersDataset):
    def __init__(
        self,
        model: Module,
        seqlen: int,
        nsamples: int,
        seed: int = 0,
        split: str = "train",
        split_percent_to_use: float = 1.0,
    ):
        super().__init__(
            model=model,
            seqlen=seqlen,
            nsamples=nsamples,
            path="ptb_text_only",
            name="penn_treebank",
            seed=seed,
            split=split,
            split_percent_to_use=split_percent_to_use,
        )

        processed_data = [sample["sentence"] for sample in self._data]
        self.create_dataloader(processed_data, join_on=" ")
