from transformers import TrainingArguments as HFTrainingArgs

__all__ = ["TrainingArguments"]

@dataclass
class TrainingArguments(HFTrainingArgs):
    """
    Training arguments specific to SparseML Transformers workflow

    :param best_model_after_epoch (`int`, *optional*, defaults to None):
        The epoch after which best model will be saved; used in conjunction with `load_best_model_at_end` and
        `metric_for_best_model` training arguments
    """
    best_model_after_epoch: int = field(
        default=None,
        metadata={"help": "Epoch after which best model will be saved."},
    )

