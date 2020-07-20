"""
Base sensitivity analysis for ONNX models and grouping them for easier
use with optimization settings
"""

from typing import Tuple, Union, List, Dict, Iterable
from collections import OrderedDict
import logging
import sys
import math
import numpy
from onnx import ModelProto

from neuralmagicML.utils import bucket_iterable, clean_path
from neuralmagicML.recal import (
    KSPerfSensitivityAnalysis,
    KSLossSensitivityAnalysis,
)
from neuralmagicML.onnx.utils import check_load_model
from neuralmagicML.onnx.recal.analyzer_model import ModelAnalyzer

__all__ = [
    "check_load_perf_analysis",
    "check_load_loss_analysis",
    "optimized_performance_buckets",
    "optimized_loss_buckets",
    "optimized_balanced_buckets",
]


_LOGGER = logging.getLogger(__name__)


def check_load_perf_analysis(
    analysis: Union[
        None,
        str,
        KSPerfSensitivityAnalysis,
        Iterable[Union[str, KSPerfSensitivityAnalysis]],
    ]
) -> List[KSPerfSensitivityAnalysis]:
    """
    Check load a single or multiple perf sensitivity analysis

    :param analysis: a path or list of paths to sensitivity analysis files
        or a loaded analysis or list of analysis
    :return: a list of loaded perf sensitivity analysis or an empty list if input None
    """
    if not analysis:
        return []

    if not isinstance(analysis, Iterable):
        analysis = [analysis]

    checked = []

    for check in analysis:
        checked.append(
            KSPerfSensitivityAnalysis.load_json(clean_path(check))
            if isinstance(check, str)
            else check
        )

    return checked


def check_load_loss_analysis(
    analysis: Union[
        None,
        str,
        KSLossSensitivityAnalysis,
        Iterable[Union[str, KSLossSensitivityAnalysis]],
    ]
) -> List[KSLossSensitivityAnalysis]:
    """
    Check load a single or multiple loss sensitivity analysis

    :param analysis: a path or list of paths to sensitivity analysis files
        or a loaded analysis or list of analysis
    :return: a list of loaded loss sensitivity analysis or an empty list if input None
    """
    if not analysis:
        return []

    if not isinstance(analysis, Iterable):
        analysis = [analysis]

    checked = []

    for check in analysis:
        checked.append(
            KSLossSensitivityAnalysis.load_json(clean_path(check))
            if isinstance(check, str)
            else check
        )

    return checked


def optimized_performance_buckets(
    model: Union[str, ModelProto, ModelAnalyzer],
    analysis: Union[
        str, KSPerfSensitivityAnalysis, Iterable[Union[str, KSPerfSensitivityAnalysis]],
    ],
    num_buckets: int = 3,
    bottom_percent: float = 0.05,
) -> Dict[str, Tuple[int, float]]:
    """
    Group the prunable layers from a models performance analysis into buckets
    based on how much they affect the performance of the model.
    This is based on the sparse_comparison metric from the perf analysis.
    So, to measure the effect on performance calculates the speedup over baseline
    for a given node in the model at a set sparsity.

    :param model: the loaded model, a file path to the onnx model,
        or a ModelAnalyzer instance for the model used to retrieve prunable layers
    :param analysis: the performance analysis as a path or list of paths
        to sensitivity analysis files or a loaded analysis or list of analysis,
        if multiple are supplied then will take the average between them
    :param num_buckets: the number of base buckets to group the results into
    :param bottom_percent: the bottom percentage of layers for affecting performance
        the least to group into a separate bucket
    :return: the assigned buckets and scores for each prunable node,
        a lower bucket means the node affects the performance less
        and generally should be pruned less
    """
    analyzer = (
        model
        if isinstance(model, ModelAnalyzer)
        else ModelAnalyzer(check_load_model(model))
    )
    analysis = check_load_perf_analysis(analysis)

    if not analysis:
        raise ValueError("performance analysis required for creating buckets")

    bucketed_scores = OrderedDict()

    for perf in analysis:
        for node in perf.results:
            if not node.id_:
                _LOGGER.debug("skipping node, no id: {}".format(node))
                continue

            node_analyzer = analyzer.get_node(node.id_)

            if not node_analyzer:
                _LOGGER.warning(
                    "could not find node with id {} in ModelAnalyzer, skipping".format(
                        node.id_
                    )
                )

            if not node_analyzer.prunable:
                _LOGGER.debug("skipping node, not prunable: {}".format(node))
                continue

            if node.id_ not in bucketed_scores:
                bucketed_scores[node.id_] = []

            # normalize by batch size to keep the timings at an equivalent per item
            bucketed_scores[node.id_].append(node.sparse_comparison() / perf.batch_size)

    buckets = bucket_iterable(
        [(key, numpy.mean(val)) for key, val in bucketed_scores.items()],
        num_buckets,
        bottom_percent,
        sort_highest=True,
        sort_key=lambda v: v[1],
    )  # type: List[Tuple[int, Tuple[str, float]]]
    bucket_ids = OrderedDict([(v[1][0], (v[0], v[1][1])) for v in buckets])

    return bucket_ids


def optimized_loss_buckets(
    model: Union[str, ModelProto, ModelAnalyzer],
    analysis: Union[
        str, KSLossSensitivityAnalysis, Iterable[Union[str, KSLossSensitivityAnalysis]],
    ],
    num_buckets: int = 3,
    top_percent: float = 0.05,
) -> Dict[str, Tuple[int, float]]:
    """
    Group the prunable layers from a models loss analysis into buckets
    based on how much they affect the loss of the model.
    This is based on the spase_integral metric from the loss analysis.
    So, to measure the effect on loss calculates the total effect on the loss
    for set pruning levels.

    :param model: the loaded model, a file path to the onnx model,
        or a ModelAnalyzer instance for the model used to retrieve prunable layers
    :param analysis: the loss analysis as a path or list of paths
        to sensitivity analysis files or a loaded analysis or list of analysis,
        if multiple are supplied then will take the average between them
    :param num_buckets: the number of base buckets to group the results into
    :param top_percent: the top percentage of layers for affecting loss the most
        to group into a separate bucket
    :return: the assigned buckets and scores for each prunable node,
        a lower bucket means the node affects the loss more
        and generally should be pruned less
    """
    analyzer = (
        model
        if isinstance(model, ModelAnalyzer)
        else ModelAnalyzer(check_load_model(model))
    )
    analysis = check_load_loss_analysis(analysis)

    if not analysis:
        raise ValueError("loss analysis required for creating buckets")

    bucketed_scores = OrderedDict()

    for loss in analysis:
        for node in loss.results:
            if not node.id_:
                _LOGGER.debug("skipping node, no id: {}".format(node))
                continue

            node_analyzer = analyzer.get_node(node.id_)

            if not node_analyzer:
                _LOGGER.warning(
                    "could not find node with id {} in ModelAnalyzer, skipping".format(
                        node.id_
                    )
                )

            if not node_analyzer.prunable:
                _LOGGER.debug("skipping node, not prunable: {}".format(node))
                continue

            if node.id_ not in bucketed_scores:
                bucketed_scores[node.id_] = []

            bucketed_scores[node.id_].append(node)

    bucketed_scores = [
        (key, numpy.mean([v.sparse_integral for v in val]).item())
        for key, val in bucketed_scores.items()
    ]

    buckets = bucket_iterable(
        [val for val in bucketed_scores],
        num_buckets,
        top_percent,
        sort_highest=True,
        sort_key=lambda v: v[1],
    )  # type: List[Tuple[int, Tuple[str, float]]]
    bucket_ids = OrderedDict([(v[1][0], (v[0], v[1][1])) for v in buckets])

    return bucket_ids


def optimized_balanced_buckets(
    model: Union[str, ModelProto, ModelAnalyzer],
    perf_analysis: Union[
        str, KSPerfSensitivityAnalysis, Iterable[Union[str, KSPerfSensitivityAnalysis]],
    ],
    loss_analysis: Union[
        str, KSLossSensitivityAnalysis, Iterable[Union[str, KSLossSensitivityAnalysis]],
    ],
    num_buckets: int = 3,
    edge_percent: float = 0.05,
) -> Dict[str, Tuple[int, float, float]]:
    """
    Group the prunable layers from a models perf and loss analysis into buckets
    based on how much they affect both for the model.
    Desired table is the following for the defaults:

    |              Perf Bot 5   Perf Low   Perf Med   Perf High
    | Loss Top 5   -1           -1         -1         -1
    | Loss High    -1           0          0          1
    | Loss Med     -1           0          1          2
    | Loss Low     -1           1          2          2

    :param model: the loaded model, a file path to the onnx model,
        or a ModelAnalyzer instance for the model used to retrieve prunable layers
    :param perf_analysis: the performance analysis as a path or list of paths
        to sensitivity analysis files or a loaded analysis or list of analysis
    :param loss_analysis: the loss analysis as a path or list of paths
        to sensitivity analysis files or a loaded analysis or list of analysis
    :param num_buckets: the number of base buckets to group the results into
    :param edge_percent: the edge percentage of layers for affecting perf the least
        or the loss the most to group into a separate bucket
    :return: the assigned buckets and perforrmance, loss scores
        for each prunable node, a lower number means
        the node affects the performance less and/or the loss more
        and generally should be pruned less
    """
    analyzer = (
        model
        if isinstance(model, ModelAnalyzer)
        else ModelAnalyzer(check_load_model(model))
    )

    perf_buckets = optimized_performance_buckets(
        analyzer, perf_analysis, num_buckets, edge_percent
    )
    loss_buckets = optimized_loss_buckets(
        analyzer, loss_analysis, num_buckets, edge_percent
    )
    balanced_buckets = OrderedDict()
    perf_ids = set(perf_buckets.keys())
    loss_ids = set(loss_buckets.keys())

    if len(perf_ids - loss_ids) > 0:
        _LOGGER.warning(
            "could not find perf nodes with ids {} in the loss_analysis".format(
                list(perf_ids - loss_ids)
            )
        )

    if len(loss_ids - perf_ids) > 0:
        _LOGGER.warning(
            "could not find loss nodes with ids {} in the perf_analysis".format(
                list(loss_ids - perf_ids)
            )
        )

    for id_, (perf_bucket, perf_value) in perf_buckets.items():
        if id_ not in loss_buckets:
            continue

        loss_bucket, loss_value = loss_buckets[id_]

        if perf_bucket == -1 or loss_bucket == -1:
            # one of the worst performers for perf or loss, keep in bottom bucket
            balanced_buckets[id_] = (-1, perf_value, loss_value)
        else:
            # assign to the floor() of the average of the buckets between perf and loss
            # if the average is close to max bucket, then round up
            bucket_avg = (perf_bucket + loss_bucket) / 2.0 + sys.float_info.epsilon
            bucket = (
                math.floor(bucket_avg)
                if bucket_avg < num_buckets - 1.5
                else num_buckets - 1
            )
            balanced_buckets[id_] = (bucket, perf_value, loss_value)

    return balanced_buckets
