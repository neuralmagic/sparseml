import torch

from sparseml.transformers import SparseAutoModelForCausalLM, oneshot


# define a sparseml recipe for GPTQ W8A8 quantization
recipe = """
quant_stage:
    quant_modifiers:
        GPTQModifier:
            sequential_update: false
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: "int"
                        symmetric: true
                        strategy: "tensor"
                    input_activations:
                        num_bits: 8
                        type: "int"
                        symmetric: true
                        dynamic: true
                        strategy: "token"
                    targets: ["Linear"]
"""

# setting device_map to auto to spread the model evenly across all available GPUs
# load the model in as bfloat16 to save on memory and compute
model_stub = "zoo:llama2-7b-ultrachat200k_llama2_pretrain-base"
model = SparseAutoModelForCausalLM.from_pretrained(
    model_stub, torch_dtype=torch.bfloat16, device_map="auto"
)

# uses SparseML's built-in preprocessing for ultra chat
dataset = "ultrachat-200k"

# save location of quantized model out
output_dir = "./output_llama7b_w8a8_dynamic_compressed"

# set dataset config parameters
splits = {"calibration": "train_gen[:5%]"}
max_seq_length = 512
pad_to_max_length = False
num_calibration_samples = 512

# apply recipe to the model and save quantized output in an int8 compressed format
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    output_dir=output_dir,
    splits=splits,
    max_seq_length=max_seq_length,
    pad_to_max_length=pad_to_max_length,
    num_calibration_samples=num_calibration_samples,
    save_compressed=True,
)
