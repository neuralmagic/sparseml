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

from typing import List, Optional, Union

from sparseml.transformers import SparseAutoModelForCausalLM, SparseAutoTokenizer


try:
    import numpy
    import torch
    from datasets import Dataset as HuggingFaceDataset
    from datasets import load_dataset
    from torch.nn import CrossEntropyLoss
    from tqdm import tqdm
except ImportError as err:
    raise ImportError(
        "perplexity evaluation requires the following packages to be installed: "
        "datasets, numpy, torch, tqdm, transformers kindly install these packages "
        "using `pip install sparseml[transformers, torch]`"
    ) from err

from sparseml.evaluation.registry import SparseMLEvaluationRegistry
from sparsezoo.evaluation.results import Dataset, Evaluation, Metric, Result


@SparseMLEvaluationRegistry.register("perplexity")
def perplexity_eval(
    model_path,
    datasets: str = "wikitext",
    batch_size: int = 1,
    device: Optional[str] = None,
    limit: Optional[int] = None,
    **kwargs,
) -> Result:
    """
    Perform perplexity evaluation on a language model.

    :param model_path: The path to the model to evaluate
    :param datasets: The name of the dataset to evaluate on
    :param batch_size: The batch size to use for evaluation
    :param device: The device to use for evaluation
    :param limit: The number of samples to evaluate on
    :param kwargs: Additional arguments for the evaluation
    """
    if isinstance(datasets, list):
        raise NotImplementedError(
            "Running perplexity evaluation on multiple datasets is not supported"
        )
    dataset_config_name = _infer_dataset_config_name(datasets)
    task = "text-generation"
    split = kwargs.pop("split", None)
    model = SparseAutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = SparseAutoTokenizer.from_pretrained(model_path)

    input_text = _load_perplexity_dataset(
        dataset_name=datasets,
        dataset_config_name=dataset_config_name,
        split=split,
        limit=limit,
        text_column_name=kwargs.pop("text_column_name", None),
    )
    add_start_token = True
    max_length = None

    # Adapted from
    # https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py#L103

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

    # if batch_size is 1, set the pad token to be the eos token
    if batch_size == 1:
        tokenizer.pad_token = tokenizer.eos_token

    # fetch tokenized inputs and attention masks
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
            # prepend <BOS> token tensor to each input encoding and
            # attention mask
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

        # calculate perplexity for each batch
        perplexity_batch = torch.exp(
            (
                loss_fct(shift_logits.transpose(1, 2), shift_labels)
                * shift_attention_mask_batch
            ).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    mean_ppl = numpy.mean(ppls)
    raw = {"mean_perplexity": mean_ppl}

    # wrap the perplexity result in a Result object
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


def _infer_dataset_config_name(datasets: str):
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
    text_column_name: Union[str, List[str], None] = None,
    split: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[str]:
    """
    Loads the dataset for perplexity evaluation.

    :param dataset_name: The name of the dataset to load
    :param dataset_config_name: The name of the dataset config to load
    :param text_column_name: The name of the column containing the text data
        if None, defaults to "text". If a list of column names is passed, the
        columns will be concatenated to form the input text
    :param split: The split of the dataset to load, if None uses test split
        if available, otherwise uses train split
    :param nsamples: The number of samples to load from the dataset
    :return: The loaded dataset as a list of strings
    """

    dataset: HuggingFaceDataset = _fetch_dataset_split(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        split=split,
    )
    text_column_name: List[str] = _verify_text_column_name(
        dataset=dataset, text_column_name=text_column_name
    )

    inputs = []
    for sample in dataset:
        input_sample = "".join(sample[column_name] for column_name in text_column_name)
        if input_sample != "":
            inputs.append(input_sample)
        if limit is not None and len(inputs) >= limit:
            break
    return inputs


def _fetch_dataset_split(
    dataset_name: str, dataset_config_name: Optional[str] = None, split=None
):
    """
    Loads and returns the specified split of the dataset.

    :param dataset_name: The name of the dataset to load from the HuggingFace
        datasets library
    :param dataset_config_name: The name of the dataset config to load, if any.
    :param split: The split of the dataset to load, if None uses test split
        if available, otherwise uses train split. Also supports HuggingFace
        style splits such as "train[:10%]", "test", "validation", etc.
    :return: The loaded dataset split
    """
    dataset = load_dataset(dataset_name, dataset_config_name, split=split)
    if split is not None:
        # specified split was found in the dataset
        return dataset

    # try to infer the split to use
    if "test" in dataset:
        return dataset["test"]

    if "train" in dataset:
        return dataset["train"]

    raise ValueError(
        f"Neither 'test' nor 'train' split found in dataset {dataset_name}. "
        "Specify a valid split using the 'split' argument."
    )


def _verify_text_column_name(
    dataset: HuggingFaceDataset,
    text_column_name: Union[str, List[str], None] = None,
) -> List[str]:
    """
    Verifies that the dataset contains the specified text column name(s),
    and returns the text column name(s) to use for evaluation as a list.

    :param dataset: The huggingface dataset to verify
    :param text_column_name: The name of the column containing the text data
        if None, defaults to "text". If a list of column names is passed, all
        columns must be present in the dataset
    :return: The text column name(s) to use for evaluation as a list of strings
    """
    text_column_names = text_column_name or ["text"]

    if isinstance(text_column_names, str):
        text_column_names = [text_column_name]

    for column_name in text_column_names:
        if column_name not in dataset.column_names:
            raise ValueError(
                f"Dataset {dataset} does not contain a column named {column_name}"
            )
    return text_column_names
