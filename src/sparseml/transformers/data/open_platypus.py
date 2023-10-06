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


@TransformersDataset.register(name="open_platypus")
class OpenPlatypus(TransformersDataset):
    ALPACA_TEMPLATE = {
        "prompt_input": "Below is an instruction that describes a task, paired with an "
        "input that provides further context. Write a response that appropriately "
        "completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n"
        "{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a "
        "response that appropriately completes the request.\n\n### Instruction:\n{"
        "instruction}\n\n### Response:\n",
    }

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
            path="garage-bAInd/Open-Platypus",
            name=None,
            seed=seed,
            split=split,
            use_max_tokens=False,
            split_percent_to_use=split_percent_to_use,
        )

        processed_data = []
        for sample in self._data:
            if "input" in sample:
                processed_sample = self.ALPACA_TEMPLATE["prompt_input"].format(
                    instruction=sample["instruction"], input=sample["input"]
                )
            else:
                processed_sample = self.ALPACA_TEMPLATE["prompt_no_input"].format(
                    instruction=sample["instruction"]
                )

            if "output" in sample:
                processed_sample += sample["output"]
            processed_data.append(processed_sample)

        self.create_dataloader(processed_data)
