from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/tiny_starcoder_py"
device="cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0]))