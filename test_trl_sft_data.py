from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from sparseml.transformers import (
    SFTTrainer,
    TrainingArguments, 
    SparseAutoModelForCausalLM, 
    SparseAutoTokenizer
)

dataset = load_dataset("gsm8k", "main", split="train")
model_path = "neuralmagic/Llama-2-7b-pruned50-retrained"
output_dir = "./output_trl_sft_test_7b_gsm8k_sft_data"
model = SparseAutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = SparseAutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

recipe = """
test_stage:
  pruning_modifiers:
    ConstantPruningModifier:
      targets: ['re:.*q_proj.weight', 're:.*k_proj.weight', 're:.*v_proj.weight', 're:.*o_proj.weight',
        're:.*gate_proj.weight', 're:.*up_proj.weight', 're:.*down_proj.weight']
      start: 0
"""


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"Question: {example['question'][i]}\n Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

response_template = "Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
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
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    args=training_args,
    max_seq_length=512
)
trainer.train()
trainer.save_model(output_dir=trainer.args.output_dir)