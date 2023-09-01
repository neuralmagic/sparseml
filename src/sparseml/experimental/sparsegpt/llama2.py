import torch
import torch.nn as nn

from sparseml.experimental.sparsegpt.layer_compressor import BaseCompressor
from sparseml.experimental.sparsegpt.model_preprocessor import QuantizationModelPreprocessor
from sparseml.experimental.sparsegpt.sequential import SequentialSparseGPT
from sparseml.experimental.sparsegpt.utils import catch, execute_offloaded_module


class SequentialSparseGPT_Llama2(SequentialSparseGPT):
    def compressible_layers(self):
        return self.model.model.layers


class Llama2BottomCompressor(BaseCompressor):
    """
    Llama2 specific
    """

    def post_compress(
        self,
        dataloader=None,
        nsamples: int = None,
        device: str = "cuda:0",
        **kwargs,
    ):
        cached_inputs = cache_attention_mask(self.model, dataloader, device, nsamples)

        outputs = execute_offloaded_module(
            self.model.model.embed_tokens,
            dataloader,
            device,
            nsamples,
            overwrite_buffer=False,
        )
        torch.cuda.empty_cache()

        extras = {"outputs": outputs}.update(cached_inputs)
        return self.model, extras


def prepare_sparsegpt(model, dataloader, args) -> SequentialSparseGPT:
    model_preprocessors = []
    if args.recipe:
        model_preprocessors.append(
            QuantizationModelPreprocessor(
                args.recipe,
                dataloader,
                args.observer_batches,
                llam2_eval,
            )
        )
    bottom_compressor = Llama2BottomCompressor(model)
    sequential_sparsegpt = SequentialSparseGPT_Llama2(
        model,
        recipe=args.recipe,
        model_preprocessors=model_preprocessors,
        bottom_compressor=bottom_compressor,
    )

    return sequential_sparsegpt


def load_model(args):
    model = args.model

    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
    seqlen = model.config.max_position_embeddings
    return model, seqlen


def load_data(args, seqlen, split=0.1):
    name = args.dataset
    nsamples = args.nsamples
    model = args.model
    seed = args.seed

    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    elif "platypus" in name:
        return get_openplatypus(nsamples, seed, seqlen, model, split)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc, tokenizer


def get_openplatypus(nsamples, seed, seqlen, model, split):
    from datasets import load_dataset

    traindata = load_dataset("garage-bAInd/Open-Platypus", split="train")

    import random

    random.seed(seed)
    traindata = list(traindata)
    random.shuffle(traindata)
    number_test_samples = max(1, int(split * len(traindata)))
    testdata = traindata[-number_test_samples:]
    traindata = traindata[:-number_test_samples]
    if nsamples is not None and nsamples > len(traindata):
        traindata = traindata[:nsamples]

    alpaca_template = {
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    }

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)

    def _process_sample(sample):
        if "input" in sample:
            processed_sample = alpaca_template["prompt_input"].format(instruction=sample["instruction"], input=sample["input"])
        else:
            processed_sample = alpaca_template["prompt_no_input"].format(instruction=sample["instruction"])

        if "output" in sample:
            processed_sample += sample["output"]

        tokenized_sample = tokenizer(
            processed_sample,
            truncation=True,
            max_length=seqlen,
            return_tensors="pt",
            padding=False,
        )["input_ids"][0]

        if tokenized_sample[-1] != tokenizer.eos_token_id:
            if len(tokenized_sample) == seqlen:
                tokenized_sample[-1] = tokenizer.eos_token_id
            else:
                tokenized_sample = torch.concatenate(
                    (tokenized_sample, torch.tensor((tokenizer.eos_token_id,))),
                )

        return tokenized_sample

    trainenc = [_process_sample(sample) for sample in traindata]
    testenc = [_process_sample(sample) for sample in testdata]

    return trainenc, testenc, tokenizer


def cache_attention_mask(model, data_loader, device, nsamples):
    model.model.embed_tokens.to(device)
    model.model.layers[0].to(device)
    cached_inputs = catch(model, model.model.layers[0], "attention_mask", data_loader, nsamples)
    model.model.embed_tokens.cpu()
    model.model.layers[0].cpu()
    torch.cuda.empty_cache()
    return cached_inputs


def llam2_eval(model, data_loader, device, nsamples=None):
    offloaded_model = OffLoadedModule(model)
    for sample in data_loader:
        offloaded_model(sample)

    # Catch attention mask
    cached_inputs = cache_attention_mask(model, data_loader, device, nsamples)

    buffer = execute_offloaded_module(
        model.model.embed_tokens,
        data_loader,
        device,
        nsamples,
        overwrite_buffer=False,
    )
    for layer in model.model.layers:
        buffer = execute_offloaded_module(
            layer,
            buffer,
            device,
            cached_inputs=cached_inputs,
            use_cache=False,
        )

    del cached_inputs
    torch.cuda.empty_cache()

    buffer = execute_offloaded_module(
        model.model.norm,
        buffer,
        device,
    )
    logits = execute_offloaded_module(
        model.lm_head,
        buffer,
        device,
    )

    return logits


@torch.no_grad()
def perplexity_eval(
        model,
        testenc,
        dev,
        nsamples,
        dataset: str,
        log_wandb: bool = False,
):
    print("Evaluating perplexity...")

    logits = llam2_eval(model, testenc, dev, nsamples)

    neg_log_likelihood = 0.
    number_tokens = 0
    for label, logit in logits:
        shift_logits = logit[:-1, :].contiguous()
        shift_labels = label[1:].to(dev)
        neg_log_likelihood += nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        number_tokens += shift_labels.numel()

    ppl = torch.exp(neg_log_likelihood / number_tokens)
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})
