import torch
from sparseml.transformers import oneshot, SparseAutoModelForCausalLM, SparseAutoTokenizer
from sparseml.transformers.finetune.data.data_args import DataTrainingArguments
from sparseml.transformers.finetune.data import TextGenerationDataset
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator, AutoModelForCausalLM
from sparseml.pytorch.utils import tensors_to_device
from sparsetensors.quantization.utils import is_module_quantized
import math 

def old_quant_linear():
    return """
    test_stage:
        quant_modifiers:
            QuantizationModifier:
                ignore:
                    - LlamaRotaryEmbedding
                    - LlamaRMSNorm
                    - SiLU
                    - MatMulLeftInput_QK
                    - MatMulRightInput_QK
                    - MatMulOutput_QK
                    - MatMulLeftInput_PV
                    - MatMulRightInput_PV
                    - MatMulOutput_PV
                    - Embedding
                scheme_overrides:
                    Linear:
                        weights:
                            num_bits: 8
                            symmetric: true
                            strategy: "tensor"
                        input_activations: null
                        output_activations: null
    """

def new_quant_linear():
    return """
    test_stage:
        quant_modifiers:
            vLLMQuantizationModifier:
                ignore: []
                config_groups:
                    group_0:
                        weights:
                            num_bits: 8
                            type: "int"
                            symmetric: true
                            strategy: "tensor"
                        input_activations: null
                        output_activations: null
                        targets: ["Linear"]
    """

def labeled_dataloader(dataset_name, model_name):
    tokenizer = SparseAutoTokenizer.from_pretrained(model_name)
    data_args = DataTrainingArguments(
        dataset=dataset_name,
        max_seq_length=512,
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
        calib_dataset, batch_size=1, collate_fn=DefaultDataCollator()
    )

    return data_loader

def run_oneshot(model, recipe, dataset):
    num_calibration_samples = 8
    max_seq_length = 512
    pad_to_max_length = False

    oneshot(
        model=model,
        dataset=dataset,
        overwrite_output_dir=True,
        max_seq_length = max_seq_length,
        num_calibration_samples=num_calibration_samples,
        recipe=recipe,
        pad_to_max_length=pad_to_max_length
    )

def test_quantization_eval():
    num_comparisons = 4
    model_stub = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    model_old = SparseAutoModelForCausalLM.from_pretrained(model_stub, device_map="cuda:0")
    dataset = "open_platypus"
    run_oneshot(model_old, old_quant_linear(), dataset)

    model_new = AutoModelForCausalLM.from_pretrained(model_stub, device_map="cuda:1")
    run_oneshot(model_new, new_quant_linear(), dataset)

    old_quant_count = 0
    old_info = {}
    for name, module in model_old.named_modules():
        if hasattr(module, "weight_fake_quant"):
            old_info[name] = (module.weight_fake_quant.scale.item(), module.weight_fake_quant.zero_point.item())
            old_quant_count += 1

    new_quant_count = 0
    new_info = {}
    for name, module in model_new.named_modules():
        if is_module_quantized(module):
            new_info[name] = (module.weight_scale.item(), module.weight_zero_point.item())
            new_quant_count += 1

    assert old_quant_count == new_quant_count
    for name, (o_scale, o_zp) in old_info.items():
        n_scale, n_zp = new_info[name]
        if not math.isclose(o_scale, n_scale, abs_tol=1e-4, rel_tol=1e-4):
            print(f"mismatch {name} {o_scale} {n_scale}")
        if not math.isclose(o_zp, n_zp, rel_tol=1e-3):
            print(f"mismatch {name} {o_zp} {n_zp}")

    dataloader = labeled_dataloader(dataset, model_stub)
    total_old_ppl = 0.0
    total_new_ppl = 0.0
    for idx, sample in enumerate(dataloader):
        if idx >= num_comparisons:
            return
        old_output = model_old(**(tensors_to_device(sample, "cuda:0")))
        new_output = model_new(**(tensors_to_device(sample, "cuda:1")))
        old_ppl = torch.exp(old_output.loss)
        new_ppl = torch.exp(new_output.loss)
        print(f"Perplexity: new {new_ppl} old {old_ppl}")
        total_old_ppl += old_ppl
        total_new_ppl += new_ppl
        del old_output
        del new_output
        torch.cuda.empty_cache()


    avg_new_ppl = total_new_ppl / num_comparisons
    avg_old_ppl = total_old_ppl / num_comparisons
    print(f"Avg Perplexity: new {avg_new_ppl} old {avg_old_ppl}")

test_quantization_eval()