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
from pathlib import Path

import pytest
import torch


try:
    from clearml import Task

    is_clearml = True
except Exception:
    is_clearml = False

from sparseml.transformers import train


@pytest.mark.skipif(not is_clearml, reason="clearML not installed")
def test_finetune_wout_recipe(tmp_path: Path):
    recipe_str = None
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if not torch.cuda.is_available():
        device = "cpu"
    dataset = "open_platypus"
    concatenate_data = False
    output_dir = tmp_path
    max_steps = 50
    splits = "train"

    Task.init(project_name="test", task_name="test_oneshot_and_finetune")

    train(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        recipe=recipe_str,
        max_steps=max_steps,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
    )
