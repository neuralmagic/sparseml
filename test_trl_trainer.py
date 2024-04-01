from sparseml.transformers import SparseAutoModelForCausalLM, SparseAutoTokenizer
from sparseml.transformers.finetune.sft_trainer import SFTTrainer
from transformers import DefaultDataCollator
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data import TextGenerationDataset
from peft import LoraConfig

model_path = "facebook/opt-350m"
output_dir = "./output_trl_sft_test"

model = SparseAutoModelForCausalLM.from_pretrained(model_path)
tokenizer = SparseAutoTokenizer.from_pretrained(model_path)

data_args = DataTrainingArguments(dataset = "open_platypus")
dataset_manager = TextGenerationDataset.load_from_registry(
    data_args.dataset,
    data_args=data_args,
    split="train",
    tokenizer=tokenizer,
)
raw_dataset = dataset_manager.get_raw_dataset()
train_dataset = dataset_manager.tokenize_and_process(raw_dataset)
print(f"--> Training Set Length = {len(train_dataset)}")


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

data_collator = DefaultDataCollator()
trainer = SFTTrainer(
    model=model,
    model_state_path=model_path,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    peft_config=lora_config,
    dataset_text_field="text"
)
trainer.train()