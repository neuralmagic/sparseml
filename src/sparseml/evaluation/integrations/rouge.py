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

from typing import Optional

import evaluate
from sparseml.evaluation.registry import SparseMLEvaluationRegistry


@SparseMLEvaluationRegistry.register("rouge")
def rouge_score(
    target: str,
    datasets: str = "cnn-daily-mail",
    batch_size: int = 1,
    device: str = "cuda",
    nsamples: Optional[int] = None,
    **kwargs,
):

    # initialize the model

    # initialize the dataset

    # get predictions and references

    # compute the rouge score

    rouge = evaluate.load("rouge")
    predictions = ...  # ["hello there", "general kenobi"]
    references = ...  # ["hello there", "general kenobi"]
    results = rouge.compute(predictions=predictions, references=references)
    return results
