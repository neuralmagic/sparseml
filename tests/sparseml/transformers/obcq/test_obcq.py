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

from sparseml.modifiers.obcq.utils.helpers import ppl_eval_general
from sparseml.transformers.data import TransformersDataset
from sparseml.transformers.sparsification.obcq.obcq import one_shot
from sparseml.transformers.sparsification.obcq.utils.helpers import llama_forward


def test_obcq_tinystories():
    tiny_model_path = "Xenova/llama2.c-stories15M"
    device = "cuda:0"

    # test recipe with 50% sparsity, quantization and smoothquant
    tiny_model = one_shot(
        model_path=tiny_model_path,
        dataset_name="open_platypus",
        num_samples=64,
        device=device,
        recipe_file="tests/sparseml/transformers/obcq/test_tiny.yaml",
    )

    dataset = TransformersDataset.load_from_registry(
        "wikitext2",
        model=tiny_model_path,
        seqlen=tiny_model.seqlen,
        nsamples=64,
        seed=0,
        split="test",
    )
    test_data = dataset.loader
    perplexity = ppl_eval_general(
        llama_forward, tiny_model, test_data, device, max_samples_per_iteration=8
    )

    # we aren't expecting good results from this tiny model, but this should catch any
    # egregious errors with the OBCQ algorithm
    assert perplexity < 10000.0
