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


import argparse
import json
import os

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

import evaluate
import nltk
from accelerate import Accelerator
from lm_eval.utils import stop_sequences_criteria


nltk.download("punkt")


ARTICLE_TEMPLATE = "Article:\n{article}"

SUMMARY_TEMPLATE = "\n\n### Summarization:\n"


def load_model(model_path):
    return AutoModelForCausalLM.from_pretrained(model_path)


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tokenizer


def postprocess_text(preds, labels, first_k_preds):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)[:first_k_preds]) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def main(model_path, batch, dataset_path, dataset_name):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)

    accelerator = Accelerator() if args.use_accelerate else None

    with accelerator.main_process_first():
        dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split="validation")
        if args.samples > 0:
            dataset = dataset.shuffle(seed=42).select(range(args.samples))

        result_path = os.path.join(model_path, args.output_dir)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

    if args.generation == "lm-eval-harness":
        # Similar to the default decoding strategy used by
        # lm-evaluation-harness
        gen_kwargs = {
            "do_sample": False,
            "temperature": 1.0,  # To disable warning
            "top_p": 1.0,  # To disable warning
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "max_new_tokens": 512,
        }
    elif args.generation == "top_k":
        # Similar to GPT-2 decoding strategy used for summarization
        # (see their paper, section 3.6)
        gen_kwargs = {
            "do_sample": True,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        }
    else:
        raise ValueError(f"Unknown decoding strategy: {args.generation}")

    def _process_sample(sample):
        article = ARTICLE_TEMPLATE.format(article=sample["article"])
        tok_summary = tokenizer(SUMMARY_TEMPLATE)

        # Exclude the BOS from the tokenized summary
        tok_summary = {k: tok_summary[k][1:] for k in tok_summary}

        max_tok_article = args.max_input_length - len(tok_summary["input_ids"])
        tok_article = tokenizer(
            article, max_length=max_tok_article, truncation=True, padding="max_length"
        )

        model_inputs = {k: tok_article[k] + tok_summary[k] for k in tok_article}

        prompt_length = len(model_inputs["input_ids"])
        highlights = tokenizer(
            sample["highlights"],
            max_length=prompt_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["tok_highlights"] = highlights["input_ids"]

        # Using "label" for sample ID since it will be recognized and reserved by
        # the default data collator used below
        model_inputs["label"] = hash(sample["id"])

        return model_inputs

    tokenized_dataset = dataset.map(_process_sample, batched=False, num_proc=16)
    remove_columns = dataset.column_names
    tokenized_dataset = tokenized_dataset.remove_columns(remove_columns)
    tokenized_dataset.set_format("torch")

    data_collator = default_data_collator
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        collate_fn=data_collator,
    )
    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)

    if accelerator.is_main_process:
        saved_preds = {"ids": [], "predictions": [], "highlights": []}
        rouge_score = evaluate.load("rouge")

    model.eval()
    for step, batch in enumerate(tqdm(dataloader)):
        labels = batch["labels"]
        with torch.no_grad():
            if args.generation == "lm-eval-harness":
                stop = ["\n\n", "Article:"]
                initial_decoder_input_length = batch["input_ids"].shape[1]
                batch_size = batch["input_ids"].shape[0]
                stopping_criteria = stop_sequences_criteria(
                    tokenizer, stop, initial_decoder_input_length, batch_size
                )
            else:
                stopping_criteria = None

            prompt_length = batch["input_ids"].shape[1]
            if args.use_accelerate:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    stopping_criteria=stopping_criteria,
                    **gen_kwargs,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                highlights = batch["tok_highlights"]

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                highlights = accelerator.gather(highlights).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()
            else:
                # Code path for debugging only with 1 GPU
                batch = {k: batch[k].to(model.device) for k in batch.keys()}
                generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    stopping_criteria=stopping_criteria,
                    **gen_kwargs,
                )
                highlights = batch["tok_highlights"]

                generated_tokens = generated_tokens.cpu().numpy()
                highlights = highlights.cpu().numpy()
                labels = labels.cpu().numpy()
                batch = None
                torch.cuda.empty_cache()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            generated_summary_tokens = generated_tokens[:, prompt_length:]
            decoded_preds = tokenizer.batch_decode(
                generated_summary_tokens, skip_special_tokens=True
            )
            decoded_highlights = tokenizer.batch_decode(
                highlights, skip_special_tokens=True
            )
            decoded_preds, decoded_highlights = postprocess_text(
                decoded_preds, decoded_highlights, args.first_k_preds
            )

            assert len(labels) == len(decoded_preds) == len(decoded_highlights)

            if accelerator.is_main_process:
                saved_preds["ids"] += labels.tolist()
                saved_preds["predictions"] += decoded_preds
                saved_preds["highlights"] += decoded_highlights

    if accelerator.is_main_process:
        results = rouge_score.compute(
            predictions=saved_preds["predictions"], references=saved_preds["highlights"]
        )
        print(f"Rouge score: {results}")

        with open(os.path.join(result_path, f"predictions.json"), "w") as f:
            json.dump(saved_preds, f)

        result_file_name = (
            f"rouge_{args.samples}samples.json"
            if args.samples > 0
            else f"rouge_full_validation.json"
        )
        results.update(
            {
                "generation": args.generation,
                "generation_config": gen_kwargs,
                "prompt": ARTICLE_TEMPLATE + SUMMARY_TEMPLATE,
            }
        )
        result_file_path = os.path.join(result_path, result_file_name)
        assert not os.path.exists(
            result_file_path
        ), f"File {result_file_path} already exists! Results will not be saved."
        with open(result_file_path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ROUGE score")
    parser.add_argument(
        "--use-accelerate",
        type=bool,
        default=False,
        help="Use accelerate. Default: False",
    )
    parser.add_argument("--model-path", type=str, help="model path")
    parser.add_argument(
        "--output-dir", type=str, default="rouge", help="Output directory"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Max new tokens"
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=2048,
        help="Max tokenized input length to model",
    )
    parser.add_argument(
        "--first-k-preds", type=int, default=-1, help="Use first K predictions"
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--samples", type=int, default=-1, help="Numer of samples. Default to all."
    )
    parser.add_argument(
        "--generation",
        type=str,
        default="lm-eval-harness",
        help="Generation strategies: lm-eval-harness, top_k",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="top_k in the top_k stategy"
    )
    parser.add_argument(
        "--dataset-path", type=str, default="cnn_dailymail", help="dataset path"
    )
    parser.add_argument(
        "--dataset-name", type=str, default="3.0.0", help="dataset name"
    )

    args = parser.parse_args()

    main(args.model_path, args.batch, args.dataset_path, args.dataset_name)
