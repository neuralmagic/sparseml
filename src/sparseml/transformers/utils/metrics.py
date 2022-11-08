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
    # predictions and targets shape: (num_samples, num_labels)
    predictions = predictions.astype(bool)
    targets = targets.astype(bool)

    # compute per-class TP, FP, FN
    true_positives = numpy.logical_and(predictions, targets).sum(axis=0)
    false_positives = numpy.logical_and(predictions, ~targets).sum(axis=0)
    false_negatives = numpy.logical_and(~predictions, targets).sum(axis=0)

    # compute per-class precision, recall, f1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    # labels with no TP/FN will evaluate to nan - convert to 0.0
    precision = numpy.nan_to_num(precision)
    recall = numpy.nan_to_num(recall)
    f1 = numpy.nan_to_num(f1)

    # compile results into required str -> float dict
    results = {}
    for idx in range(predictions.shape[1]):
        label = id_to_label[idx] if id_to_label else str(idx)  # default to str idx

        results[f"precision_{label}"] = precision[idx]
        results[f"recall_{label}"] = recall[idx]
        results[f"f1_{label}"] = f1[idx]

    # add macro averages to results
    results["macro_average_precision"] = precision.mean()
    results["macro_average_recall"] = recall.mean()
    results["macro_average_f1"] = f1.mean()

    return results
