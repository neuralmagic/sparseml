from sparseml.transformers import SparseAutoModelForCausalLM, SparseAutoTokenizer, oneshot
from copy import deepcopy
import torch

model_name = "Qwen/Qwen1.5-MoE-A2.7B"

model = SparseAutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda:0",
    torch_dtype=torch.float16,
)
og_model = deepcopy(model)
tokenizer = SparseAutoTokenizer.from_pretrained(
    model_name
)

dataset = "open-platypus"
recipe = "tests/sparseml/transformers/compression/recipes/new_quant_full.yaml"

oneshot(
        model=model,
        dataset=dataset,
        overwrite_output_dir=True,
        output_dir="./output_one_shot",
        recipe=recipe,
        num_calibration_samples=8
        
    )

prompt = "Why did the transformer cross the road?"
prompt_tokenized = tokenizer(prompt, return_tensors="pt").to(model.device)
print('----')
print(f"Output from the original model: {tokenizer.decode(og_model.generate(**prompt_tokenized, max_length=50)[0])}")
print('----')
tokenizer = SparseAutoTokenizer.from_pretrained("./output_one_shot")
prompt_tokenized = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"Output from the quantized model: {tokenizer.decode(model.generate(**prompt_tokenized, max_length=50)[0])}")
print('----')
model = SparseAutoModelForCausalLM.from_pretrained("./output_one_shot", device_map="cuda:1", torch_dtype=torch.float16)
print(f"Output from the quantized model (reloaded): {tokenizer.decode(model.generate(**prompt_tokenized.to(model.device), max_length=50)[0])}")
print('----')