# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
import unittest
from pathlib import Path

import pytest

from tests.testing_utils import requires_torch


@pytest.mark.integration
@requires_torch
class TestSafetensors(unittest.TestCase):
    def setUp(self):
        self.output = Path("./finetune_output")

    def test_safetensors(self):
        import torch

        from sparseml.transformers import train

        model = "Xenova/llama2.c-stories15M"
        device = "cuda:0"
        if not torch.cuda.is_available():
            device = "cpu"

        dataset = "open_platypus"
        output_dir = self.output / "output1"
        max_steps = 10
        splits = {"train": "train[:10%]"}

        train(
            model=model,
            dataset=dataset,
            output_dir=output_dir,
            max_steps=max_steps,
            splits=splits,
            oneshot_device=device,
        )

        assert os.path.exists(output_dir / "model.safetensors")
        assert not os.path.exists(output_dir / "pytorch_model.bin")

        # test we can also load
        new_output_dir = self.output / "output2"
        train(
            model=output_dir,
            dataset=dataset,
            output_dir=new_output_dir,
            max_steps=max_steps,
            splits=splits,
            oneshot_device=device,
        )

    def tearDown(self):
        shutil.rmtree(self.output)
