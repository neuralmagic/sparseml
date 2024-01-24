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
# flake8: noqa: F401


import os

import pytest

from sparseml.evaluation.integrations.perplexity import perplexity_eval


def perplexity_dependencies_available():
    try:
        import datasets
        import numpy
        import torch
        import transformers
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.skipif(
    not perplexity_dependencies_available(),
    reason="Skipping perplexity tests due to missing dependencies",
)
def test_perplexity_against_huggingface():
    target = "Xenova/llama2.c-stories15M"
    batch_size = 4
    nsamples = 100
    dataset = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"
    split = "test"

    # run through the evaluation
    actual = round(
        perplexity_eval(
            target=target,
            batch_size=batch_size,
            nsamples=nsamples,
            datasets=dataset,
        ).raw["mean_perplexity"],
        2,
    )

    # compare to the huggingface evaluation
    expected = _huggingface_perplexity_eval(
        dataset=dataset,
        dataset_config_name=dataset_config_name,
        split=split,
        target=target,
        batch_size=batch_size,
        nsamples=nsamples,
    )

    assert actual == expected


def _huggingface_perplexity_eval(
    dataset, dataset_config_name, split, target, batch_size, nsamples
):
    import numpy
    import torch
    import tqdm
    from datasets import load_dataset
    from torch.nn import CrossEntropyLoss
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # _compute() is taken as is from
    # https://github.com/huggingface/evaluate/metrics/perplexity/perplexity.py

    def _compute(
        predictions,
        model_id,
        batch_size: int = 16,
        add_start_token: bool = True,
        device=None,
        max_length=None,
    ):

        if device is not None:
            assert device in [
                "gpu",
                "cpu",
                "cuda",
            ], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(
                tokenizer.special_tokens_map_extended.values()
            )
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use"
            " for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. "
            "Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 1)
            ), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two "
            "tokens long. Run with add_start_token=True if inputting strings of "
            "only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in tqdm.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)
                ).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [
                        torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(
                            device
                        ),
                        attn_mask,
                    ],
                    dim=1,
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (
                    loss_fct(shift_logits.transpose(1, 2), shift_labels)
                    * shift_attention_mask_batch
                ).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": numpy.mean(ppls)}

    input_texts = load_dataset(dataset, dataset_config_name, split=split)["text"]
    input_texts = [s for s in input_texts if s != ""][:nsamples]
    results = _compute(model_id=target, predictions=input_texts, batch_size=batch_size)
    return round(results["mean_perplexity"], 2)
