__all__ = [
    "INSTRUCTION_SETS",
    "INFERENCE_ENGINE_TYPES",
    "OPTIM_MODIFIER_TYPES",
    "PRUNING_LOSS_ESTIMATION_TYPES",
    "PRUNING_STRUCTURE_TYPES",
    "QUANTIZATION_LEVELS",
    "LR_CLASSES",
    "FILE_SOURCES",
    "MODEL_DATA_SOURCES",
]

INSTRUCTION_SETS = ["AVX2", "AVX512", "VNNI"]
INFERENCE_ENGINE_TYPES = ["neural_magic", "ort_cpu", "ort_gpu"]

OPTIM_MODIFIER_TYPES = ["pruning", "quantization", "lr_schedule", "trainable"]

PRUNING_LOSS_ESTIMATION_TYPES = ["weight_magnitude", "one_shot"]
PRUNING_STRUCTURE_TYPES = ["unstructured", "block_2", "block_4", "channel", "filter"]

QUANTIZATION_LEVELS = ["int8", "int16"]

LR_CLASSES = ["set", "step", "multi_step", "exponential"]

FILE_SOURCES = ["uploaded", "generated"]
MODEL_DATA_SOURCES = ["uploaded", "downloaded_path", "downloaded_repo"]
