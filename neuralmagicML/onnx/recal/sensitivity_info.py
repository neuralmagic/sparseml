"""
Sensitivity and analysis info for ONNX models
"""

from typing import Tuple, Union, List, Dict, Iterable, Any
from collections import OrderedDict
import logging
import uuid
import csv
import json
from onnx import ModelProto

from neuralmagicML.utils import clean_path, create_parent_dirs
from neuralmagicML.recal import (
    KSPerfSensitivityAnalysis,
    KSLossSensitivityAnalysis,
    KSSensitivityResult,
)
from neuralmagicML.onnx.utils import check_load_model
from neuralmagicML.onnx.recal.analyzer_model import ModelAnalyzer, NodeAnalyzer
from neuralmagicML.onnx.recal.sensitivity import (
    check_load_perf_analysis,
    check_load_loss_analysis,
    optimized_performance_buckets,
    optimized_loss_buckets,
)

__all__ = ["SensitivityNodeInfo", "SensitivityModelInfo"]


_LOGGER = logging.getLogger(__name__)


class SensitivityNodeInfo(object):
    """
    Container object for handling the sensitivity info for an ONNX node.
    Includes info from analyzers, perf and loss KS sensitivity,
    and perf and loss KS sensitivity buckets.

    :param analyzer: the ONNX analyzer for the node this object is being created for,
        if any
    """

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]):
        """
        Create new sensitivity node info from a dictionary of values.
        Expected to match the format as given in the dict() call.

        :param dictionary: the dictionary to create a result out of
        :return: the created KSLossSensitivityResult
        """
        analyzer = NodeAnalyzer(
            model=None,
            node=None,
            **dictionary["analyzer"] if dictionary["analyzer"] else None
        )
        node_info = SensitivityNodeInfo(analyzer)

        for (analysis_id, analysis) in dictionary["perf_results"]:
            node_info.add_perf_result(
                KSSensitivityResult.from_dict(analysis), analysis_id
            )

        for (analysis_id, analysis) in dictionary["loss_results"]:
            node_info.add_loss_result(
                KSSensitivityResult.from_dict(analysis), analysis_id
            )

        node_info.set_perf_bucket(dictionary["perf_bucket"])
        node_info.set_loss_bucket(dictionary["loss_bucket"])

        return node_info

    def __init__(
        self, analyzer: Union[NodeAnalyzer, None],
    ):
        self._analyzer = analyzer
        self._perf_results = OrderedDict()  # type: Dict[str, KSSensitivityResult]
        self._loss_results = OrderedDict()  # type: Dict[str, KSSensitivityResult]
        self._perf_bucket = None  # type: Union[None, Tuple[int, float]]
        self._loss_bucket = None  # type: Union[None, Tuple[int, float]]

    @property
    def analyzer(self) -> Union[NodeAnalyzer, None]:
        """
        :return: the ONNX analyzer for the node this object is being created for,
            if any
        """
        return self._analyzer

    @property
    def perf_results(self) -> Dict[str, KSSensitivityResult]:
        """
        :return: the performance sensitivity results for the node and the
            associated ID for the parent sensitivity analysis
        """
        return self._perf_results

    @property
    def loss_results(self) -> Dict[str, KSSensitivityResult]:
        """
        :return: the loss sensitivity results for the node and the
            associated ID for the parent sensitivity analysis
        """
        return self._loss_results

    @property
    def perf_bucket(self) -> Union[Tuple[int, float], None]:
        """
        :return: the performance sensitivity bucket and score for the node
        """
        return self._perf_bucket

    @property
    def loss_bucket(self) -> Union[Tuple[int, float], None]:
        """
        :return: the loss sensitivity bucket and score for the node
        """
        return self._loss_bucket

    def dict(self) -> Dict[str, Any]:
        """
        :return: dictionary representation of the current instance
        """

        return {
            "analyzer": self.analyzer.dict() if self.analyzer else None,
            "perf_results": [
                (key, val.dict()) for key, val in self.perf_results.items()
            ],
            "loss_results": [
                (key, val.dict()) for key, val in self.loss_results.items()
            ],
            "perf_bucket": self.perf_bucket,
            "loss_bucket": self.loss_bucket,
        }

    def add_perf_result(self, result: KSSensitivityResult, analysis_id: str):
        """
        add a performance sensitivity result for the node

        :param result: the performance sensitivity result for the node
        :param analysis_id: the id for the performance sensitivity analysis
        """
        self._perf_results[analysis_id] = result

    def remove_perf_result(self, analysis_id: str, strict: bool = False):
        """
        remove a previous performance result for the node

        :param analysis_id: the id for the performance sensitivity analysis
        :param strict: True to raise an error if not found, False otherwise
        """
        if analysis_id in self._perf_results:
            del self._perf_results[analysis_id]
        elif strict:
            raise ValueError(
                "analysis_id {} not found in perf_results".format(analysis_id)
            )

    def add_loss_result(self, result: KSSensitivityResult, analysis_id: str):
        """
        add a loss sensitivity result for the node

        :param result: the loss sensitivity result for the node
        :param analysis_id: the id for the loss sensitivity analysis
        """
        self._loss_results[analysis_id] = result

    def remove_loss_result(self, analysis_id: str, strict: bool = False):
        """
        remove a previous loss result for the node

        :param analysis_id: the id for the loss sensitivity analysis
        :param strict: True to raise an error if not found, False otherwise
        """
        if analysis_id in self._loss_results:
            del self._loss_results[analysis_id]
        elif strict:
            raise ValueError(
                "analysis_id {} not found in loss_results".format(analysis_id)
            )

    def set_perf_bucket(self, bucket: Union[Tuple[int, float], None]):
        self._perf_bucket = bucket

    def set_loss_bucket(self, bucket: Union[Tuple[int, float], None]):
        self._loss_bucket = bucket


class SensitivityModelInfo(object):
    """
    Container object for handling the sensitivity info for an ONNX model.
    Includes info from analyzers, perf and loss KS sensitivity,
    and perf and loss KS sensitivity buckets.

    :param analyzer: the ONNX analyzer for the model this object is being created for
    """

    @staticmethod
    def load_json(path: str):
        """
        :param path: the path to load a previous analysis from
        :return: the ModelAnalyzer instance from the json
        """
        path = clean_path(path)

        with open(path, "r") as file:
            objs = json.load(file)

        return ModelAnalyzer.from_dict(objs)

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]):
        """
        :param dictionary: the dictionary to create an analysis object from
        :return: the SensitivityModelInfo instance created from the dictionary
        """
        analyzer = (
            ModelAnalyzer.from_dict(dictionary["analyzer"])
            if dictionary["analyzer"]
            else None
        )
        model_info = SensitivityModelInfo(analyzer)
        model_info._nodes = OrderedDict(
            [
                (key, NodeAnalyzer(model=None, node=None, **val))
                for (key, val) in dictionary["nodes"]
            ]
        )
        model_info._perf_analysis = OrderedDict(
            [
                (key, KSPerfSensitivityAnalysis.from_dict(val))
                for (key, val) in dictionary["perf_analysis"]
            ]
        )
        model_info._loss_analysis = OrderedDict(
            [
                (key, KSPerfSensitivityAnalysis.from_dict(val))
                for (key, val) in dictionary["loss_analysis"]
            ]
        )
        model_info._perf_buckets = (
            OrderedDict(dictionary["perf_buckets"])
            if dictionary["perf_buckets"]
            else None
        )
        model_info._loss_buckets = (
            OrderedDict(dictionary["loss_buckets"])
            if dictionary["loss_buckets"]
            else None
        )

        return model_info

    @staticmethod
    def from_sensitivities(
        model: Union[str, ModelProto, ModelAnalyzer],
        perf_analysis: Union[
            None,
            str,
            KSPerfSensitivityAnalysis,
            Iterable[Union[str, KSPerfSensitivityAnalysis]],
        ],
        loss_analysis: Union[
            None,
            str,
            KSLossSensitivityAnalysis,
            Iterable[Union[str, KSLossSensitivityAnalysis]],
        ],
        num_buckets: int = 3,
        edge_percent: float = 0.05,
    ):
        """
        Convenience wrapper to create sensitivity model info from model info along with
        performance and loss sensitivities

        :param model: the loaded model, a file path to the onnx model,
            or a ModelAnalyzer instance for the model
        :param perf_analysis: the performance analysis as a path or list of paths
            to sensitivity analysis files or a loaded analysis or list of analysis
        :param loss_analysis: the loss analysis as a path or list of paths
            to sensitivity analysis files or a loaded analysis or list of analysis
        :param num_buckets: the number of base buckets to group the results into
        :param edge_percent: the edge percentage of layers for affecting perf the least
            or the loss the most to group into a separate bucket
        :return: the SensitivityModelInfo instance created from the sensitivities
        """
        perf_analysis = check_load_perf_analysis(perf_analysis)
        loss_analysis = check_load_loss_analysis(loss_analysis)

        analyzer = (
            model
            if isinstance(model, ModelAnalyzer)
            else ModelAnalyzer(check_load_model(model))
        )
        model_info = SensitivityModelInfo(analyzer)

        for analysis in perf_analysis:
            model_info.add_perf_analysis(analysis)

        for analysis in loss_analysis:
            model_info.add_loss_analysis(analysis)

        if perf_analysis:
            perf_buckets = optimized_performance_buckets(
                analyzer, perf_analysis, num_buckets, edge_percent
            )
            model_info.set_perf_buckets(perf_buckets)

        if loss_analysis:
            loss_buckets = optimized_loss_buckets(
                analyzer, loss_analysis, num_buckets, edge_percent
            )
            model_info.set_loss_buckets(loss_buckets)

        return model_info

    def __init__(self, analyzer: Union[None, ModelAnalyzer]):
        self._analyzer = analyzer
        self._nodes = OrderedDict()  # type: Dict[str, SensitivityNodeInfo]
        self._perf_analysis = (
            OrderedDict()
        )  # type: Dict[str, KSPerfSensitivityAnalysis]
        self._loss_analysis = (
            OrderedDict()
        )  # type: Dict[str, KSLossSensitivityAnalysis]
        self._perf_buckets = None  # type: Union[None, Dict[str, Tuple[int, float]]]
        self._loss_buckets = None  # type: Union[None, Dict[str, Tuple[int, float]]]

        for node in self._analyzer.nodes:
            if not node.prunable:
                continue

            self._nodes[node.id_] = SensitivityNodeInfo(node)

    @property
    def analyzer(self) -> Union[None, ModelAnalyzer]:
        """
        :return: the ONNX analyzer for the model this object is created for
        """
        return self._analyzer

    @property
    def nodes(self) -> Dict[str, SensitivityNodeInfo]:
        """
        :return: a dictionary mapping node id or name to the sensitivity node
            info that has been collected so far for that node
        """
        return self._nodes

    @property
    def perf_analysis(self) -> Dict[str, KSPerfSensitivityAnalysis]:
        """
        :return: a dictionary mapping performance sensitivity analysis that have
            been added to unique ids for the analysis generated when each was added
        """
        return self._perf_analysis

    @property
    def loss_analysis(self) -> Dict[str, KSLossSensitivityAnalysis]:
        """
        :return: a dictionary mapping loss sensitivity analysis that have
            been added to unique ids for the analysis generated when each was added
        """
        return self._loss_analysis

    @property
    def perf_buckets(self) -> Union[Dict[str, Tuple[int, float]], None]:
        """
        :return: the performance buckets that have been set for the model, if any
        """
        return self._perf_buckets

    @property
    def loss_buckets(self) -> Union[Dict[str, Tuple[int, float]], None]:
        """
        :return: the loss buckets that have been set for the model, if any
        """
        return self._loss_buckets

    def dict(self) -> Dict[str, Any]:
        """
        :return: dictionary representation of the current instance
        """
        return {
            "analyzer": self.analyzer.dict(),
            "nodes": [(key, val.dict()) for key, val in self.nodes.items()],
            "perf_analysis": [
                (key, val.dict()) for key, val in self.perf_analysis.items()
            ],
            "loss_analysis": [
                (key, val.dict()) for key, val in self.loss_analysis.items()
            ],
            "perf_buckets": [(key, val) for key, val in self.perf_buckets.items()]
            if self.perf_buckets
            else None,
            "loss_buckets": [(key, val) for key, val in self.loss_buckets.items()]
            if self.loss_buckets
            else None,
        }

    def table(self) -> List[List[Any]]:
        """
        :return: a rectangular table representation of the current instance,
            useful for storing in csv or spreadsheets
        """
        return [
            self.table_headers(),
            self.table_titles(),
            *self.table_node_rows(),
        ]

    def table_headers(self) -> List[str]:
        """
        :return: a list of headers (groupings of titles) for a table
            representation of the current instance
        """
        headers = [
            "Node Info",  # Index
            "",  # Id / Name
            "",  # Op Type
            "",  # Weight Name
            "",  # Bias Name
            "Bucket Info",  # Loss Bucket
            "",  # Loss Score
            "",  # Perf Bucket
            "",  # Perf Score
            "Node Extended Info",  # Attributes
            "",  # Param Count
            "",  # Prunable Param Count
            "",  # Zeroed Prunable Param Count
            "",  # Sparsity
            "",  # Input Shapes
            "",  # Output Shapes
            "",  # Weight Shape
            "",  # flops
        ]

        for index, _ in enumerate(self.loss_analysis.keys()):
            headers.extend(
                [
                    "Loss Analysis {}".format(index),  # Baseline
                    "",  # Integrated Total
                    "",  # 90% Sparsity Comparison
                ]
            )

        for index, perf_analysis in enumerate(self.perf_analysis.values()):
            headers.extend(
                [
                    "Perf Analysis {} num_cores: {} batch_size: {}".format(
                        index, perf_analysis.num_cores, perf_analysis.batch_size
                    ),  # Model Baseline Total (sec)
                    "",  # Baseline (sec)
                    "",  # Baseline Percent
                    "",  # Integrated Total (sec)
                    "",  # 90% Sparsity Comparison (sec)
                    "",  # Sparse Averages (sec)
                ]
            )

        return headers

    def table_titles(self) -> List[str]:
        """
        :return: a list of titles for a table representation of the current instance
        """
        titles = [
            "Index",
            "Id / Name",
            "Op Type",
            "Weight Name",
            "Bias Name",
            "Loss Bucket",
            "Loss Score",
            "Perf Bucket",
            "Perf Score",
            "Attributes",
            "Param Count",
            "Prunable Param Count",
            "Zeroed Prunable Param Count",
            "Sparsity",
            "Input Shapes",
            "Output Shapes",
            "Weight Shape",
            "flops",
        ]

        for _ in self.loss_analysis.keys():
            titles.extend(["Baseline", "Integrated Total", "90% Sparsity Comparison"])

        for _ in self.perf_analysis.keys():
            titles.extend(
                [
                    "Model Baseline Total (sec)",
                    "Baseline (sec)",
                    "Baseline Percent",
                    "Integrated Total (sec)",
                    "90% Sparsity Comparison (sec)",
                    "Sparse Averages (sec)",
                ]
            )

        return titles

    def table_node_rows(self) -> List[Any]:
        """
        :return: a list of rows for a table representation of all the
            nodes for the current instance
        """
        return [self.table_node_row(id_) for id_ in self.nodes.keys()]

    def table_node_row(self, id_: str) -> List[Any]:
        """
        :param id_: the id of the node to get a row for
        :return: a list of attributes for a table representation for a given node
            that matches the id
        """
        node = self._nodes[id_]
        row = [
            # Index
            list(self.nodes.keys()).index(id_),
            # Id / Name
            id_,
            # Op type
            node.analyzer.op_type if node.analyzer else None,
            # Weight Name
            node.analyzer.weight_name if node.analyzer else None,
            # Bias Name
            node.analyzer.bias_name if node.analyzer else None,
            # Loss Bucket
            node.loss_bucket[0] if node.loss_bucket else None,
            # Loss Score
            node.loss_bucket[1] if node.loss_bucket else None,
            # Perf Bucket
            node.perf_bucket[0] if node.perf_bucket else None,
            # Perf Score
            node.perf_bucket[1] if node.perf_bucket else None,
            # Attributes
            node.analyzer.attributes if node.analyzer else None,
            # Param Count
            node.analyzer.params if node.analyzer else None,
            # Prunable Params Count
            node.analyzer.prunable_params
            if node.analyzer
            else None,
            # Zeroed Prunable Param Count
            node.analyzer.prunable_params_zeroed
            if node.analyzer
            else None,
            # Sparsity
            node.analyzer.prunable_params_zeroed / node.analyzer.prunable_params
            if node.analyzer
            and node.analyzer.prunable_params_zeroed is not None
            and node.analyzer.prunable_params
            else None,
            # Input Shapes
            node.analyzer.input_shapes if node.analyzer else None,
            # Output Shapes
            node.analyzer.output_shapes if node.analyzer else None,
            # Weight Shape
            node.analyzer.weight_shape if node.analyzer else None,
            # flops
            node.analyzer.flops if node.analyzer else None,
        ]

        for analysis_id, loss_analysis in self.loss_analysis.items():
            node_result = (
                node.loss_results[analysis_id]
                if analysis_id in node.loss_results
                else None
            )
            row.extend(
                [
                    # Baseline
                    node_result.baseline_average
                    if node_result is not None and node_result.has_baseline
                    else None,
                    # Integrated Total
                    node_result.sparse_integral if node_result is not None else None,
                    # 90% Sparsity Comparison
                    node_result.sparse_comparison()
                    if node_result is not None
                    else None,
                ]
            )

        for analysis_id, perf_analysis in self.perf_analysis.items():
            node_result = (
                node.perf_results[analysis_id]
                if analysis_id in node.perf_results
                else None
            )
            row.extend(
                [
                    # Model Baseline Total (sec)
                    perf_analysis.results_model.baseline_average
                    if perf_analysis.results_model.has_baseline
                    else None,
                    # Baseline (sec)
                    node_result.baseline_average
                    if node_result is not None and node_result.has_baseline
                    else None,
                    # Baseline Percent
                    node_result.baseline_average
                    / perf_analysis.results_model.baseline_average
                    if node_result is not None
                    and node_result.has_baseline
                    and perf_analysis.results_model.has_baseline
                    else None,
                    # Integrated Total (sec)
                    node_result.sparse_integral if node_result is not None else None,
                    # 90% Sparsity Comparison (sec)
                    node_result.sparse_comparison()
                    if node_result is not None
                    else None,
                    # Sparse Averages (sec)
                    json.dumps(node_result.averages)
                    if node_result is not None
                    else None,
                ]
            )

        return row

    def save_json(self, path: str):
        """
        :param path: the path to save the json file at representing the layer
            sensitivities
        """
        if not path.endswith(".json"):
            path += ".json"

        path = clean_path(path)
        create_parent_dirs(path)

        with open(path, "w") as file:
            json.dump(self.dict(), file, indent=2)

    def save_csv(self, path: str):
        """
        :param path: the path to save the json file at representing the layer
            sensitivities
        """
        if not path.endswith(".csv"):
            path += ".csv"

        path = clean_path(path)
        create_parent_dirs(path)

        with open(path, "w") as file:
            writer = csv.writer(
                file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )

            for row in self.table():
                writer.writerow(row)

    def add_perf_analysis(self, analysis: KSPerfSensitivityAnalysis) -> str:
        """
        Add a performance analysis for the current model to the info

        :param analysis: the performance analysis to add
        :return: the created id used to reference the added analysis
        """
        analysis_id = uuid.uuid4().hex
        self._perf_analysis[analysis_id] = analysis

        for result in analysis.results:
            node = None

            if result.id_ is not None and result.id_ in self._nodes:
                node = self._nodes[result.id_]
            elif result.name and result.has_baseline and result.name in self._nodes:
                node = self._nodes[result.name]
            elif (
                result.id_ is not None
                and self._analyzer is not None
                and self._analyzer.get_node(result.id_) is not None
            ):
                # found a node not marked as prunable, add it in
                node = SensitivityNodeInfo(self._analyzer.get_node(result.id_))
                self._nodes[result.id_] = node
            elif result.id_ is not None:
                _LOGGER.warning(
                    (
                        "could not find perf result with node id {} in nodes, "
                        "creating new node info with no onnx info"
                    ).format(result.id_)
                )
                node = SensitivityNodeInfo(None)
                self._nodes[result.id_] = node
            elif result.name and result.has_baseline:
                # performance node from baseline run that didn't map to onnx
                # keep these in so baseline timing totals are reflected appropriately
                _LOGGER.warning(
                    (
                        "found baseline perf result with no node id and name {}, "
                        "creating new node info with no onnx info"
                    ).format(result.name)
                )
                node = SensitivityNodeInfo(None)
                self._nodes[result.name] = node
            else:
                _LOGGER.warning(
                    (
                        "found non baseline perf result with no node id "
                        "and name {}, skipping"
                    ).format(result.name)
                )

            if node is not None:
                node.add_perf_result(result, analysis_id)

        return analysis_id

    def add_loss_analysis(self, analysis: KSLossSensitivityAnalysis) -> str:
        """
        Add a loss analysis for the current model to the info

        :param analysis: the loss analysis to add
        :return: the created id used to reference the added analysis
        """
        analysis_id = uuid.uuid4().hex
        self._loss_analysis[analysis_id] = analysis

        for result in analysis.results:
            node = None

            if result.id_ is not None and result.id_ in self._nodes:
                node = self._nodes[result.id_]
            elif result.name and result.name in self._nodes:
                node = self._nodes[result.name]
            elif (
                result.id_ is not None
                and self._analyzer is not None
                and self._analyzer.get_node(result.id_) is not None
            ):
                # found a node not marked as prunable, add it in
                node = SensitivityNodeInfo(self._analyzer.get_node(result.id_))
                self._nodes[result.id_] = node
            elif result.id_ is not None:
                _LOGGER.warning(
                    (
                        "could not find loss result with node id {} in nodes, "
                        "creating new node info with no onnx info"
                    ).format(result.id_)
                )
                node = SensitivityNodeInfo(None)
                self._nodes[result.id_] = node
            elif result.name:
                # loss node that didn't map to onnx
                # keep these in so user can map
                _LOGGER.warning(
                    (
                        "found loss result with no node id and name {}, "
                        "creating new node info with no onnx info"
                    ).format(result.name)
                )
                node = SensitivityNodeInfo(None)
                self._nodes[result.name] = node
            else:
                _LOGGER.warning(
                    (
                        "found non baseline loss result with no node id "
                        "or name, skipping"
                    ).format(result.name)
                )

            if node is not None:
                node.add_loss_result(result, analysis_id)

        return analysis_id

    def set_perf_buckets(self, buckets: Dict[str, Tuple[int, float]]):
        """
        Set the performance buckets for the current model to the info

        :param buckets: the buckets to set
        """
        self._perf_buckets = buckets

        for node_id, bucket in buckets.items():
            if node_id in self._nodes:
                self._nodes[node_id].set_perf_bucket(bucket)
            else:
                _LOGGER.warning(
                    "could not find node with id {} for setting perf_bucket".format(
                        node_id
                    )
                )

    def set_loss_buckets(self, buckets: Dict[str, Tuple[int, float]]):
        """
        Set the loss buckets for the current model to the info

        :param buckets: the buckets to set
        """
        self._loss_buckets = buckets

        for node_id, bucket in buckets.items():
            if node_id in self._nodes:
                self._nodes[node_id].set_loss_bucket(bucket)
            else:
                _LOGGER.warning(
                    "could not find node with id {} for setting loss_bucket".format(
                        node_id
                    )
                )
