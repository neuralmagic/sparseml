"""
Base sensitivity analysis for ONNX models and grouping them for easier
use with optimization settings
"""


from typing import Tuple, Union, List, Dict
from collections import OrderedDict
import logging
import sys
import math
import numpy

from neuralmagicML.utils import bucket_iterable
from neuralmagicML.recal import (
    KSPerfSensitivityAnalysis,
    KSLossSensitivityAnalysis,
    KSSensitivityResult,
)
from neuralmagicML.onnx.recal.analyzer_model import ModelAnalyzer, NodeAnalyzer


__all__ = [
    "optimized_performance_buckets",
    "optimized_loss_buckets",
    "optimized_balanced_buckets",
]


def optimized_performance_buckets(
    analysis: Union[KSPerfSensitivityAnalysis, List[KSPerfSensitivityAnalysis]],
    analyzer: ModelAnalyzer,
    num_buckets: int = 3,
    bottom_percent: float = 0.05,
) -> Dict[str, Tuple[int, List[KSPerfSensitivityAnalysis]]]:
    """
    Group the prunable layers from a models performance analysis into buckets
    based on how much they affect the performance of the model.
    This is based on the sparse_comparison metric from the perf analysis.
    So, to measure the effect on performance calculates the speedup over baseline
    for a given node in the model at a set sparsity.

    :param analysis: the performance analysis, if multiple are supplied
        then will take the average between them
    :param analyzer: the ModelAnalyzer used to retrieve prunable layers
    :param num_buckets: the number of base buckets to group the results into
    :param bottom_percent: the bottom percentage of layers for affecting performance
        the least to group into a separate bucket
    :return: the assigned buckets for each prunable node, a lower number means
        the node affects the performance less and generally should be pruned less
    """
    if not analyzer:
        raise ValueError("performance analysis required for creating buckets")

    if analyzer is None:
        raise ValueError("model analyzer required for creating buckets")

    if not isinstance(analysis, list):
        analysis = [analysis]

    values = OrderedDict()

    for perf in analysis:
        for node in perf.results:
            if not node.id_:
                logging.debug("skipping node, no id: {}".format(node))
                continue

            node_analyzer = analyzer.get_node(node.id_)

            if not node_analyzer:
                logging.warning(
                    "could not find node with id {} in ModelAnalyzer, skipping".format(
                        node.id_
                    )
                )

            if not node_analyzer.prunable:
                logging.debug("skipping node, not prunable: {}".format(node))
                continue

            if node.id_ not in values:
                values[node.id_] = []

            values[node.id_].append(node)

    buckets = bucket_iterable(
        [val for val in values.values()],
        num_buckets,
        bottom_percent,
        sort_highest=True,
        sort_key=lambda vs: numpy.mean([v.sparse_comparison() for v in vs]).item(),
    )
    bucket_ids = OrderedDict([(vb[1][0].id_, vb) for vb in buckets])

    return bucket_ids


def optimized_loss_buckets(
    analysis: Union[KSLossSensitivityAnalysis, List[KSLossSensitivityAnalysis]],
    analyzer: ModelAnalyzer,
    num_buckets: int = 3,
    top_percent: float = 0.05,
) -> Dict[str, Tuple[int, List[KSLossSensitivityAnalysis]]]:
    """
    Group the prunable layers from a models loss analysis into buckets
    based on how much they affect the loss of the model.
    This is based on the spase_integral metric from the loss analysis.
    So, to measure the effect on loss calculates the total effect on the loss
    for set pruning levels.

    :param analysis: the loss analysis, if multiple are supplied
        then will take the average between them
    :param analyzer: the ModelAnalyzer used to retrieve prunable layers
    :param num_buckets: the number of base buckets to group the results into
    :param top_percent: the top percentage of layers for affecting loss the most
        to group into a separate bucket
    :return: the assigned buckets for each prunable node, a lower number means
        the node affects the loss more and generally should be pruned less
    """
    if not analysis:
        raise ValueError("loss analysis required for creating buckets")

    if analyzer is None:
        raise ValueError("model analyzer required for creating buckets")

    if not isinstance(analysis, list):
        analysis = [analysis]

    values = OrderedDict()

    for loss in analysis:
        for node in loss.results:
            if not node.id_:
                logging.debug("skipping node, no id: {}".format(node))
                continue

            node_analyzer = analyzer.get_node(node.id_)

            if not node_analyzer:
                logging.warning(
                    "could not find node with id {} in ModelAnalyzer, skipping".format(
                        node.id_
                    )
                )

            if not node_analyzer.prunable:
                logging.debug("skipping node, not prunable: {}".format(node))
                continue

            if node.id_ not in values:
                values[node.id_] = []

            values[node.id_].append(node)

    buckets = bucket_iterable(
        [val for val in values.values()],
        num_buckets,
        top_percent,
        sort_highest=True,
        sort_key=lambda vs: numpy.mean([v.sparse_integral for v in vs]).item(),
    )
    bucket_ids = OrderedDict([(vb[1][0].id_, vb) for vb in buckets])

    return bucket_ids


def optimized_balanced_buckets(
    perf_analysis: Union[KSPerfSensitivityAnalysis, List[KSPerfSensitivityAnalysis]],
    loss_analysis: Union[KSLossSensitivityAnalysis, List[KSLossSensitivityAnalysis]],
    analyzer: ModelAnalyzer,
    num_buckets: int = 3,
    edge_percent: float = 0.05,
) -> Dict[
    str, Tuple[int, List[KSPerfSensitivityAnalysis], List[KSLossSensitivityAnalysis]]
]:
    """
    Group the prunable layers from a models perf and loss analysis into buckets
    based on how much they affect both for the model.
    Desired table is the following for the defaults:

    |              Perf Bot 5   Perf Low   Perf Med   Perf High
    | Loss Top 5   -1           -1         -1         -1
    | Loss High    -1           0          0          1
    | Loss Med     -1           0          1          2
    | Loss Low     -1           1          2          2

    :param perf_analysis: the performance analysis, if multiple are supplied
        then will take the average between them
    :param loss_analysis: the loss analysis, if multiple are supplied
        then will take the average between them
    :param analyzer: the ModelAnalyzer used to retrieve prunable layers
    :param num_buckets: the number of base buckets to group the results into
    :param edge_percent: the edge percentage of layers for affecting perf the least
        or the loss the most to group into a separate bucket
    :return: the assigned buckets for each prunable node, a lower number means
        the node affects the performance less and/or the loss more
        and generally should be pruned less
    """
    perf_buckets = optimized_performance_buckets(
        perf_analysis, analyzer, num_buckets, edge_percent
    )
    loss_buckets = optimized_loss_buckets(
        loss_analysis, analyzer, num_buckets, edge_percent
    )
    balanced_buckets = OrderedDict()
    perf_ids = set(perf_buckets.keys())
    loss_ids = set(loss_buckets.keys())

    if len(perf_ids - loss_ids) > 0:
        logging.warning(
            "could not find perf nodes with ids {} in the loss_analysis".format(
                list(perf_ids - loss_ids)
            )
        )

    if len(loss_ids - perf_ids) > 0:
        logging.warning(
            "could not find loss nodes with ids {} in the perf_analysis".format(
                list(loss_ids - perf_ids)
            )
        )

    for id_, perf_bucket in perf_buckets.items():
        if id_ not in loss_buckets:
            continue

        loss_bucket = loss_buckets[id_]

        if perf_bucket[0] == -1 or loss_bucket[0] == -1:
            # one of the worst performers for perf or loss, keep in bottom bucket
            balanced_buckets[id_] = (-1, perf_bucket[1], loss_bucket[1])
        else:
            # assign to the floor() of the average of the buckets between perf and loss
            # if the average is close to max bucket, then round up
            bucket_avg = (
                perf_bucket[0] + loss_bucket[0]
            ) / 2.0 + sys.float_info.epsilon
            bucket = (
                math.floor(bucket_avg)
                if bucket_avg < num_buckets - 1.5
                else num_buckets - 1
            )
            balanced_buckets[id_] = (bucket, perf_bucket[1], loss_bucket[1])

    return balanced_buckets
