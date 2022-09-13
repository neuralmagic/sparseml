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

import pytest

from sparseml.transformers.utils.helpers import save_zoo_directory
from sparsezoo import Model


@pytest.mark.parametrize(
    "stub",
    [
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none",  # noqa E501
        "zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/pruned90_quant-none",  # noqa E501
    ],
)
def test_save_zoo_directory(stub, tmp_path_factory):
    path_to_training_outputs = tmp_path_factory.mktemp("outputs")
    save_dir = tmp_path_factory.mktemp("save_dir")

    zoo_model = Model(stub, path_to_training_outputs)
    zoo_model.download()

    zoo_model.sample_inputs.unzip()
    zoo_model.sample_outputs["framework"].unzip()

    save_zoo_directory(
        output_dir=save_dir,
        training_outputs_dir=path_to_training_outputs,
    )
    new_zoo_model = Model(str(save_dir))
    assert new_zoo_model.validate(minimal_validation=True, validate_onnxruntime=False)
