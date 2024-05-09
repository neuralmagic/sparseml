import shutil
import unittest

import pytest

from parameterized import parameterized_class
from tests.testing_utils import parse_params, requires_gpu, requires_torch


CONFIGS_DIRECTORY = "tests/sparseml/transformers/finetune/finetune_configs"
GPU_CONFIGS_DIRECTORY = "tests/sparseml/transformers/finetune/finetune_configs/gpu"


class TestOneshotAndFinetune(unittest.TestCase):
    def _test_oneshot_and_finetune(self):
        from sparseml.transformers import apply

        splits = {"train": "train[:50%]", "calibration": "train[50%:60%]"}

        apply(
            model=self.model,
            dataset=self.dataset,
            dataset_config_name=self.dataset_config_name,
            run_stages=True,
            output_dir=self.output,
            recipe=self.recipe,
            max_steps=50,
            concatenate_data=True,
            splits=splits,
            oneshot_device=self.device,
        )
    
    def tearDown(self):
        shutil.rmtree(self.output)


@pytest.mark.integration
@parameterized_class(parse_params(CONFIGS_DIRECTORY))
class TestOneshotAndFinetuneSmall(TestOneshotAndFinetune):
    model = None
    dataset = None
    recipe = None
    dataset_config_name = None 

    def setUp(self):
        import torch 
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output = "./oneshot_output"
    
    def test_oneshot_then_finetune_small(self):
        self._test_oneshot_and_finetune()


@requires_gpu
@pytest.mark.integration
@parameterized_class(parse_params(GPU_CONFIGS_DIRECTORY))
class TestOneshotAndFinetuneGPU(TestOneshotAndFinetune):
    model = None
    dataset = None
    recipe = None
    dataset_config_name = None 

    def setUp(self):
        from sparseml.transformers import SparseAutoModelForCausalLM

        self.device = "cuda:0" 
        self.output = "./oneshot_output"

        if "zoo:" in self.model:
            self.model = SparseAutoModelForCausalLM.from_pretrained(
                self.model, device_map=self.device
            )
    
    def test_oneshot_then_finetune_gpu(self):
        self._test_oneshot_and_finetune()

       

