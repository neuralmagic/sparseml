import logging
from typing import Optional

from transformers import OPTForCausalLM
from sparseml.pytorch.sparsification.obcq.fast_obcq_modifier import FastOBCQModifier
from sparseml.pytorch.sparsification.obcq.manager import RecipeManagerOneShot
from sparseml.pytorch.sparsification.obcq.data import (
    get_wikitext2,
    get_ptb,
    get_c4
)

__all__ = ["one_shot"]

_LOGGER = logging.getLogger(__name__)
SUPPORTED_DATASETS = ["wikitext2", "ptb", "c4"]


def one_shot(
    model_path: str,
    dataset_name: str,
    deploy_dir: str = ".",
    sparsity: float = 0.5,
    num_samples: int = 128,
    use_case: Optional[str] = None,
    #eval_metric: Optional[str] = None,
    quantization: Optional[bool] = False,
    num_bits: Optional[int] = 8,
    block_size: Optional[int] = 16
    #recipe_file: Optional[str] = None,
    #recipe_args: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Performs in place one shot sparsification/quantization of a model based on:

    :param model_path: path to Hugging Face stub
    :param dataset_name: Dataset to extract calibration data from
    :param deploy_dir: The output directory to save the model to
    :param sparsity: How much sparsification to apply, skipped if 0
    :param num_samples: Number of samples to extract from the dataset
    :param use_case: ML task this model targets
    :param quantization: Whether or not to apply quantization
    :param num_nits: Number of bits to quantize to if applying quantization
    """
    deploy_dir = deploy_dir / "deployment"
    if deploy_dir.exists():
        raise RuntimeError(f"deploy_dir={deploy_dir} already exists")
    
    #TODO: don't hardcode this for OPT
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.seqlen = model.config.max_position_embeddings

    data_loader_fn = None
    if dataset_name == "wikitext2":
        data_loader_fn = get_wikitext2
    elif dataset_name == "ptb":
        data_loader_fn = get_ptb
    elif dataset_name == "c4":
        data_loader_fn = get_c4
    else:
        raise ValueError(f"dataset_name={dataset_name} should be one of {SUPPORTED_DATASETS}")
    
    calibration_data, test_encoder, tokenizer = data_loader_fn(num_samples, 0, model.seqlen, model)


    modifier_args = dict(
        target_sparsity=sparsity,
        block_size=block_size,
        mse=dict(norm=2.4, grid=100, max_shrink=0.8),
        quantize=quantization,
        num_bits=num_bits,
    )
    modifier = FastOBCQModifier(**modifier_args)
    recipe = RecipeManagerOneShot([modifier])

    if recipe is not None:
        recipe.one_shot(model, calibration_data)

    #_deploy(deploy_dir, model) #TODO
    #_LOGGER.info(f" Model saved to deployment directory: {str(deploy_dir)}")
