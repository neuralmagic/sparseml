import torch
from sparseml.transformers import oneshot, SparseAutoModelForCausalLM, SparseAutoTokenizer
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data import TextGenerationDataset
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
from compressed_tensors.quantization.utils import is_module_quantized
import math

MODEL_PATH = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
MAX_SEQ_LENGTH = 512
DATASET_NAME = "open_platypus"
NUM_CALIBRATION_SAMPLES = 512
OUTPUT_PATH = "llama1.1b_new_quant_comp"

def new_quant_linear():
    return """
    test_stage:
        quant_modifiers:
            vLLMQuantizationModifier:
                ignore: ["model.layers.0.mlp.down_proj"]
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
                            symmetric: false
                            strategy: "tensor"
                        output_activations: null
                        targets: ["Linear"]
                    group_1:
                        weights:
                            num_bits: 8
                            type: "int"
                            symmetric: true
                            strategy: "tensor"
                        input_activations: null
                        output_activations: null
                        targets: ["Embedding"]
    """

def labeled_dataloader(dataset_name, model_name):
    tokenizer = SparseAutoTokenizer.from_pretrained(model_name)
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

def run_oneshot(model, recipe, dataset, output_dir):
    num_calibration_samples = NUM_CALIBRATION_SAMPLES
    max_seq_length = MAX_SEQ_LENGTH
    pad_to_max_length = False

    oneshot(
        model=model,
        dataset=dataset,
        overwrite_output_dir=True,
        output_dir=output_dir,
        max_seq_length = max_seq_length,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe,
        pad_to_max_length=pad_to_max_length,
    )

def test_quantization_reload():
    model = SparseAutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cuda:0")
    run_oneshot(model, new_quant_linear(), DATASET_NAME, OUTPUT_PATH)

    model_reloaded = SparseAutoModelForCausalLM.from_pretrained(OUTPUT_PATH, device_map="cuda:1")

    weight_info = {}
    input_info = {}
    for name, module in model.named_modules():
        if is_module_quantized(module):
            if module.quantization_scheme.weights is not None:
                weight_info[name] = (module.weight_scale.item(), module.weight_zero_point.item())
            if module.quantization_scheme.input_activations is not None:
                input_info[name] = (module.input_scale.item(), module.input_zero_point.item())

    reload_weight_info = {}
    reload_input_info = {}
    for name, module in model_reloaded.named_modules():
        if is_module_quantized(module):
            if module.quantization_scheme.weights is not None:
                reload_weight_info[name] = (module.weight_scale.item(), module.weight_zero_point.item())
            if module.quantization_scheme.input_activations is not None:
                reload_input_info[name] = (module.input_scale.item(), module.input_zero_point.item())


    for name, (o_scale, o_zp) in weight_info.items():
        n_scale, n_zp = reload_weight_info[name]
        if not o_scale == n_scale:
            print(f"weight mismatch {name} {o_scale} {n_scale}")
        if not o_zp == n_zp:
            print(f"weight mismatch {name} {o_zp} {n_zp}")


test_quantization_reload()