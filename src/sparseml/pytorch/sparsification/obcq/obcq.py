import logging
from typing import Optional

from transformers import OPTForCausalLM
from sparseml.pytorch.sparsification.obcq.manager import RecipeManagerOneShot
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
    num_samples: int = 128,
    recipe_file: Optional[str] = None,
) -> None:
    """
    Performs in place one shot sparsification/quantization of a model based on:

    :param model_path: path to Hugging Face stub
    :param dataset_name: Dataset to extract calibration data from
    :param deploy_dir: The output directory to save the model to
    :param num_samples: Number of samples to extract from the dataset
    :param recipe_file: recipe containing SparseGPT configuration
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

    recipe = RecipeManagerOneShot.from_yaml(recipe_file)
    recipe.one_shot(model, calibration_data)
