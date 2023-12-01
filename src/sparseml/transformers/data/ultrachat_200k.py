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


DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
)


@TransformersDataset.register(name="ultrachat_200k")
class Ultrachat200k(TransformersDataset):
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
            path="HuggingFaceH4/ultrachat_200k",
            name="default",
            seed=seed,
            split=split + "_sft",
            use_max_tokens=False,
            split_percent_to_use=split_percent_to_use,
        )

        if (
            not hasattr(self.tokenizer, "chat_template")
            or self.tokenizer.chat_template is None
        ):
            self.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

        processed_data = []
        for sample in self._data:
            messages = sample["messages"]
            # We add an empty system message if there is none
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})

            processed_sample = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            processed_data.append(processed_sample)

        self.create_dataloader(processed_data)
