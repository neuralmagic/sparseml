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

"""
Utilities for evaluation metric computation
"""


from typing import Dict, Optional

import numpy
from sklearn.metrics import precision_recall_fscore_support


__all__ = [
    "multi_label_precision_recall_f1",
]


def multi_label_precision_recall_f1(
    predictions: numpy.ndarray,
    targets: numpy.ndarray,
    id_to_label: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """
    computes per class and macro-averaged precision, recall, and f1 for multiple
    model sample predictions where targets may contain multiple labels

    :param predictions: array of model predictions, shape (num_samples, num_labels)
        where positive predictions are 1 and negative predictions are 0
    :param targets: array of sample targets, shape (num_samples, num_labels)
        where positive predictions are 1 and negative predictions are 0. Must
        correspond to predictions
    :param id_to_label: optional mapping of label index to string label for results
        dictionary. Will default to a string of the index
    :return: dictionary of per label and macro-average results for precision, recall,
        and f1
    """
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions)

    # compile results into required str -> float dict
    results = {}
    for idx in range(predictions.shape[1]):
        label = id_to_label[idx] if id_to_label else str(idx)  # default to str idx

        results[f"precision_{label}"] = precision[idx]
        results[f"recall_{label}"] = recall[idx]
        results[f"f1_{label}"] = f1[idx]

    # add macro averages and std to results
    results["precision_macro_average"] = precision.mean()
    results["recall_macro_average"] = recall.mean()
    results["f1_macro_average"] = f1.mean()

    results["precision_std"] = precision.std()
    results["recall_std"] = recall.std()
    results["f1_std"] = f1.std()

    return results
