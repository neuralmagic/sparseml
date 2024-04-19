import torch
from sparseml.transformers import oneshot, SparseAutoModelForCausalLM
from compressed_tensors.quantization.utils import is_module_quantized
import math
import random
import sparseml.core.session as session_manager

def old_quant_linear():
    return """
    test_stage:
        quant_modifiers:
            QuantizationModifier:
                ignore:
                    - model.layers.0.mlp.down_proj
                    - lm_head
                    - LlamaRotaryEmbedding
                    - LlamaRMSNorm
                    - SiLU
                    - MatMulLeftInput_QK
                    - MatMulRightInput_QK
                    - MatMulOutput_QK
                    - MatMulLeftInput_PV
                    - MatMulRightInput_PV
                    - MatMulOutput_PV
                scheme_overrides:
                    Linear:
                        weights:
                            num_bits: 8
                            symmetric: true
                            strategy: "tensor"
                        input_activations:
                            num_bits: 8
                            symmetric: false
                            strategy: "tensor"
                        output_activations: null
                    Embedding:
                        weights:
                            num_bits: 8
                            symmetric: true
                            strategy: "tensor"
                        input_activations: null
                        output_activations: null
            SparseGPTModifier:
                sparsity: 0.0
                block_size: 128
                sequential_update: False
                quantize: True
                targets: ["re:model.layers.\\\d+$"]
    """

def new_quant_linear():
    return """
    test_stage:
        quant_modifiers:
            vLLMQuantizationModifier:
                ignore: ["lm_head", "model.layers.0.mlp.down_proj"]
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
            SparseGPTModifier:
                sparsity: 0.0
                block_size: 128
                sequential_update: False
                quantize: True
                targets: ["re:model.layers.\\\d+$"]
    """

def run_oneshot(model, recipe, dataset, output_dir):
    num_calibration_samples = 1024
    max_seq_length = 512
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

def test_quantization_eval(input_seed):
    random.seed(input_seed)
    torch.manual_seed(input_seed)
    model_stub = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    model_old = SparseAutoModelForCausalLM.from_pretrained(model_stub, device_map="cuda:0")
    dataset = "open_platypus"
    with session_manager.create_session():
        run_oneshot(model_old, old_quant_linear(), dataset, "llama1.1b_old_quant")

    model_new = SparseAutoModelForCausalLM.from_pretrained(model_stub, device_map="cuda:1")
    with session_manager.create_session():
        run_oneshot(model_new, new_quant_linear(), dataset, "llama1.1b_new_quant")

    old_quant_count = 0
    old_quant_input_count = 0
    old_info = {}
    old_input_info = {}
    for name, module in model_old.named_modules():
        if hasattr(module, "weight_fake_quant"):
            scale = module.weight_fake_quant.scale.item()
            zp = module.weight_fake_quant.zero_point.item()
            old_info[name] = (scale, zp)
            old_quant_count += 1
        elif hasattr(module, "quant"):
            scale = module.quant.activation_post_process.scale.item()
            zp = module.quant.activation_post_process.zero_point.item()
            old_input_info[name] = (scale, zp)
            old_quant_input_count += 1

    new_quant_count = 0
    new_quant_input_count = 0
    new_info = {}
    new_input_info = {}
    for name, module in model_new.named_modules():
        if is_module_quantized(module):
            if module.quantization_scheme.weights is not None:
                new_info[name] = (module.weight_scale.item(), module.weight_zero_point.item())
                new_quant_count += 1
            if module.quantization_scheme.input_activations is not None:
                new_input_info[name] = (module.input_scale.item(), module.input_zero_point.item())
                new_quant_input_count += 1

    assert old_quant_count == new_quant_count
    assert old_quant_input_count == new_quant_input_count

    for name, (o_scale, o_zp) in old_info.items():
        if name.endswith(".module"):
            name = name[:-7]
        n_scale, n_zp = new_info[name]
        if not math.isclose(o_scale, n_scale, abs_tol=1e-3, rel_tol=1e-3):
            print(f"weight mismatch {name} {o_scale} {n_scale}")
        if not o_zp == n_zp:
            print(f"weight mismatch {name} {o_zp} {n_zp}")

    for name, (o_scale, o_zp) in old_input_info.items():
        print(name)
        n_scale, n_zp = new_input_info[name]
        if not math.isclose(o_scale, n_scale, abs_tol=1e-3, rel_tol=1e-3):
            print(f"input mismatch {name} {o_scale} {n_scale}")
        if not o_zp == n_zp:
            print(f"input mismatch {name} {o_zp} {n_zp}")

test_quantization_eval(input_seed=0)