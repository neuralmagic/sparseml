from src.sparseml.transformers.sparsification.trainer import Trainer
import logging
from transformers import AutoTokenizer
__all__ = ["export_samples"]

_LOGGER = logging.getLogger(__name__)

def export_samples(trainer: Trainer, tokenizer: AutoTokenizer, num_samples: int, real_samples = False):
    _LOGGER.info(f"Exporting {num_samples} sample inputs/outputs")
    if real_samples:
        try:
            trainer.get_eval_dataloader()
        except:
            raise ValueError("The trainer does not contain evaluation dataloader. "
                             "Either set `real_samples = False` to generate fake samples "
                             "or initialize the trainer with `eval_dataset` argument.")

    trainer.save_sample_inputs_outputs(
        num_samples_to_export=num_samples,
        tokenizer=tokenizer,
    )
    _LOGGER.info(f"{num_samples} sample inputs/outputs exported")


