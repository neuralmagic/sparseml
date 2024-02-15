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
# flake8: noqa: F401

import numpy

from evaluate import load


def dataset_mock(*args, **kwargs):
    return ["lorem ipsum", "Happy Birthday!", "Bienvenue"]


def test_perplexity_against_huggingface(monkeypatch):
    from sparseml.evaluation.integrations.perplexity import perplexity_eval

    model_path = "Xenova/llama2.c-stories15M"
    batch_size = 2
    nsamples = None
    dataset = "wikitext"

    monkeypatch.setattr(
        "sparseml.evaluation.integrations.perplexity._load_perplexity_dataset",
        dataset_mock,
    )

    # run through the evaluation
    actual = round(
        perplexity_eval(
            model_path=model_path,
            batch_size=batch_size,
            nsamples=nsamples,
            datasets=dataset,
        ).raw["mean_perplexity"],
        2,
    )

    # compare to the huggingface evaluation
    input_texts = dataset_mock()
    expected = huggingface_ppl_eval(
        predictions=input_texts, model_id=model_path, batch_size=batch_size
    )
    assert numpy.isclose(actual, expected, rtol=0.1)


def huggingface_ppl_eval(predictions, model_id, batch_size):
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(
        predictions=predictions,
        model_id=model_id,
        add_start_token=True,
        batch_size=batch_size,
    )
    return round(results["mean_perplexity"], 2)
