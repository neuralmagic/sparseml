from sparseml.transformers import SparseAutoModelForCausalLM, SparseAutoTokenizer
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from transformers import DefaultDataCollator
from torch.utils.data import DataLoader
import torch
import random
from tqdm import tqdm
from sparseml.pytorch.utils import tensors_to_device

MODEL_PATH = "llama1.1b_old_quant"
MAX_SEQ_LENGTH = 512
DATASET_NAME = "open_platypus"
NUM_COMPARISONS = 32
DEVICE = "cpu"

def get_dataloader(dataset_name, tokenizer):
    data_args = DataTrainingArguments(
        dataset=dataset_name,
        max_seq_length=MAX_SEQ_LENGTH,
        pad_to_max_length=False,
    )
    dataset_manager = TextGenerationDataset.load_from_registry(
        data_args.dataset,
        data_args=data_args,
        split="train",
        tokenizer=tokenizer,
    )
    calib_dataset = dataset_manager.tokenize_and_process(
        dataset_manager.get_raw_dataset()
    )
    data_loader = DataLoader(
        calib_dataset, 
        batch_size=1, 
        collate_fn=DefaultDataCollator(),
        sampler=torch.utils.data.RandomSampler(calib_dataset)
    )

    return data_loader

def main(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)

    model = SparseAutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE)
    tokenizer = SparseAutoTokenizer.from_pretrained(MODEL_PATH)
    dataloader = get_dataloader(DATASET_NAME, tokenizer)

    total_ppl = 0.0
    num_non_nan = 0
    for idx, sample in tqdm(enumerate(dataloader)):
        if idx >= NUM_COMPARISONS:
            break
        sample = tensors_to_device(sample, DEVICE)
        output = model(**sample)
        print(output.loss)
        if not torch.isnan(output.loss):
            ppl = torch.exp(output.loss)
            total_ppl += ppl
            num_non_nan += 1

    avg_ppl = total_ppl / num_non_nan
    print(f"Avg Perplexity over {num_non_nan} samples: {avg_ppl}")
    print(f"Ignored {NUM_COMPARISONS - num_non_nan} nans")
    return avg_ppl

if __name__ == "__main__":
    main(seed=8743)
