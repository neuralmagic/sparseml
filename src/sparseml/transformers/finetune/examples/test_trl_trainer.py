from transformers import DefaultDataCollator

from sparseml.transformers import (
    SFTTrainer,
    DataTrainingArguments, 
    TrainingArguments, 
    TextGenerationDataset, 
    SparseAutoModelForCausalLM, 
    SparseAutoTokenizer
)

model_path = "neuralmagic/Llama-2-7b-pruned50-retrained"
output_dir = "./output_trl_sft_test_7b_gsm8k"

model = SparseAutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = SparseAutoTokenizer.from_pretrained(model_path)

# Load gsm8k using SparseML dataset tools
data_args = DataTrainingArguments(dataset = "gsm8k", dataset_config_name="main", max_seq_length=512)
dataset_manager = TextGenerationDataset.load_from_registry(
    data_args.dataset,
    data_args=data_args,
    split="train",
    tokenizer=tokenizer,
)
train_dataset = dataset_manager.tokenize_and_process()
print(f"--> Training Set Length = {len(train_dataset)}")

# recipe for maintaining model sparsity during finetuning
recipe = """
test_stage:
  pruning_modifiers:
    ConstantPruningModifier:
      targets: ['re:.*q_proj.weight', 're:.*k_proj.weight', 're:.*v_proj.weight', 're:.*o_proj.weight',
        're:.*gate_proj.weight', 're:.*up_proj.weight', 're:.*down_proj.weight']
      start: 0
"""

data_collator = DefaultDataCollator()
training_args = TrainingArguments(
    output_dir=output_dir, 
    num_train_epochs=0.6, 
    logging_steps=50, 
    gradient_checkpointing=True
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    recipe=recipe,
    train_dataset=train_dataset,
    data_collator=data_collator,
    args=training_args,
    max_seq_length=data_args.max_seq_length,
    packing=True
)
trainer.train()
trainer.save_model(output_dir=trainer.args.output_dir)