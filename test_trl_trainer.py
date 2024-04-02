from transformers import DefaultDataCollator
from datasets import load_dataset

from sparseml.transformers import (
    Trainer, 
    SFTTrainer,
    DataTrainingArguments, 
    TrainingArguments, 
    TextGenerationDataset, 
    SparseAutoModelForCausalLM, 
    SparseAutoTokenizer
)

model_path = "neuralmagic/TinyLlama-1.1B-Chat-v1.0-pruned2.4"
output_dir = "./output_trl_sft_test"

model = SparseAutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = SparseAutoTokenizer.from_pretrained(model_path)

data_args = DataTrainingArguments(dataset = "open_platypus")
dataset_manager = TextGenerationDataset.load_from_registry(
    data_args.dataset,
    data_args=data_args,
    split="train",
    tokenizer=tokenizer,
)
train_dataset = dataset_manager.tokenize_and_process()
print(f"--> Training Set Length = {len(train_dataset)}")

dataset = load_dataset("imdb", split="train")

recipe = """
test_stage:
  pruning_modifiers:
    ConstantPruningModifier:
      targets: ['re:.*q_proj.weight', 're:.*k_proj.weight', 're:.*v_proj.weight', 're:.*o_proj.weight',
        're:.*gate_proj.weight', 're:.*up_proj.weight', 're:.*down_proj.weight']
      start: 0
"""

data_collator = DefaultDataCollator()
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    recipe=recipe,
    train_dataset=train_dataset,
    data_collator=data_collator,
    args=TrainingArguments(output_dir=output_dir, num_train_epochs=0.01, logging_steps=50),
    max_seq_length=data_args.max_seq_length,
    packing=True
    #dataset_text_field="text",
)
trainer.train()
trainer.save_model(output_dir=trainer.args.output_dir)