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

from typing import List, Optional


try:
    import numpy
    import torch
    from datasets import load_dataset
    from torch.nn import CrossEntropyLoss
    from tqdm import tqdm
except ImportError as err:
    raise ImportError(
        "perplexity evaluation requires the following packages to be installed: "
        "datasets, numpy, torch, tqdm kindly install these packages using "
        "`pip install sparseml[transformers, torch]`"
    ) from err

from sparseml.evaluation.registry import SparseMLEvaluationRegistry
from sparseml.evaluation.utils.helpers import fetch_recipe_path
from sparseml.transformers.integration_helper_functions import create_model
from sparsezoo.evaluation.results import Dataset, Evaluation, Metric, Result


@SparseMLEvaluationRegistry.register("perplexity")
def perplexity_eval(
    target,
    datasets: str = "wikitext",
    batch_size: int = 1,
    device: Optional[str] = None,
    nsamples: Optional[int] = None,
) -> Result:
    dataset_config_name = _infer_dataset_config_name(datasets)
    task = "text-generation"
    split = "test"
    recipe_path = fetch_recipe_path(target)
    model, other_objects = create_model(
        source_path=target,
        device=device,
        task=task,
        recipe=recipe_path,
    )
    tokenizer = other_objects["tokenizer"]
    input_text = _load_perplexity_dataset(
        dataset_name=datasets,
        dataset_config_name=dataset_config_name,
        split=split,
        nsamples=nsamples,
    )
    add_start_token = True
    max_length = None

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for "
        "padding. Please use a different model or set batch_size=1."
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
        input_text,
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

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
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
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
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

    mean_ppl = numpy.mean(ppls)
    raw = {"perplexities": ppls, "mean_perplexity": mean_ppl}

    eval = Evaluation(
        task=task,
        dataset=Dataset(
            type=task,
            name=datasets,
            config=dataset_config_name,
            split=split,
        ),
        metrics=[Metric(name="perplexity", value=mean_ppl)],
        samples=None,
    )
    return Result(formatted=[eval], raw=raw)


def _infer_dataset_config_name(datasets):
    """
    :param datasets: The name of the dataset to load
    :return: The name of the dataset config to load
    """
    if datasets == "wikitext":
        return "wikitext-2-raw-v1"
    return None


def _load_perplexity_dataset(
    dataset_name: str,
    dataset_config_name: str,
    split: str = "test",
    nsamples: Optional[int] = None,
) -> List[str]:
    """
    Loads the dataset for perplexity evaluation.

    :param dataset_name: The name of the dataset to load
    :param dataset_config_name: The name of the dataset config to load
    :param split: The split of the dataset to load
    :param nsamples: The number of samples to load from the dataset
    :return: The loaded dataset as a list of strings
    """
    dataset = load_dataset(dataset_name, dataset_config_name, split=split)["text"]
    inputs = []
    for s in dataset:
        if s != "":
            inputs.append(s)
        if nsamples is not None and len(inputs) >= nsamples:
            break
    return inputs
