from sparseml.transformers import SparseAutoModelForCausalLM, SparseAutoTokenizer
from sparseml.transformers.finetune.data import TextGenerationDataset
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from transformers import DefaultDataCollator
from torch.utils.data import DataLoader
import torch
import random
from sparseml.pytorch.utils import tensors_to_device

MODEL_PATH_OLD = "llama1.1b_old_quant_wo"
MODEL_PATH_NEW = "llama1.1b_new_quant_wo"
MAX_SEQ_LENGTH = 512
DATASET_NAME = "open_platypus"
NUM_COMPARISONS = 6

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

    model_new = SparseAutoModelForCausalLM.from_pretrained(MODEL_PATH_NEW, device_map="cuda:0")
    model_old = SparseAutoModelForCausalLM.from_pretrained(MODEL_PATH_OLD, device_map="cuda:1")
    tokenizer = SparseAutoTokenizer.from_pretrained(MODEL_PATH_NEW)
    dataloader = get_dataloader(DATASET_NAME, tokenizer)

    for idx, sample in enumerate(dataloader):
        if idx >= NUM_COMPARISONS:
            break
        sample_new = tensors_to_device(sample, "cuda:0")
        sample_old = tensors_to_device(sample, "cuda:1")
        output_new = model_new(**sample_new)
        output_old = model_old(**sample_old)
        print(torch.exp(output_new.loss).item(), torch.exp(output_old.loss).item())

if __name__ == "__main__":
    main(seed=5678)
