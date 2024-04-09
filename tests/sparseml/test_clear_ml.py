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

from clearml import Task
from sparseml.transformers import apply
from sparseml.utils import is_package_available


is_torch_available = is_package_available("torch")
if is_torch_available:
    import torch

    torch_err = None
else:
    torch = object
    torch_err = ModuleNotFoundError(
        "`torch` is not installed, use `pip install torch` to log to Weights and Biases"
    )


def test_oneshot_and_finetune(tmp_path: Path):
    recipe_str = "tests/sparseml/transformers/finetune/test_alternate_recipe.yaml"
    model = "Xenova/llama2.c-stories15M"
    device = "cuda:0"
    if is_torch_available and not torch.cuda.is_available():
        device = "cpu"
    dataset = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"
    concatenate_data = True
    run_stages = True
    output_dir = tmp_path
    max_steps = 50
    splits = {"train": "train[:50%]", "calibration": "train[50%:60%]"}

    # clearML will automatically log default capturing entries without
    # explicitly calling logger. Logs accessible in https://app.clear.ml/
    Task.init(project_name="test", task_name="test_oneshot_and_finetune")

    apply(
        model=model,
        dataset=dataset,
        dataset_config_name=dataset_config_name,
        run_stages=run_stages,
        output_dir=output_dir,
        recipe=recipe_str,
        max_steps=max_steps,
        concatenate_data=concatenate_data,
        splits=splits,
        oneshot_device=device,
    )
