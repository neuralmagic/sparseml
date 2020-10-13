"""
Helper functions and classes for flask blueprints specific to project optimizations
for pruning
"""


from typing import Union, NamedTuple, Tuple, Dict, List, Any
from collections import OrderedDict
from enum import Enum
import logging
import numpy

from neuralmagicML.utils import interpolate_list_linear


__all__ = [
    "PruningSettings",
    "PruningNodeEvaluator",
    "PruningModelEvaluator",
]


_LOGGER = logging.getLogger(__name__)


PruningSettings = NamedTuple(
    "PruningSettings",
    [
        ("mask_type", str),
        ("sparsity", Union[float, None]),
        ("balance_perf_loss", float),
        ("filter_min_sparsity", Union[float, None]),
        ("filter_min_perf_gain", Union[float, None]),
        ("filter_max_loss_drop", Union[float, None]),
    ],
)


class ValueRescaler(object):
    """
    Convenience class for normalizing / rescaling values
    """

    def __init__(self):
        self._data = []  # type: List[Tuple[float, float, float]]
        self._avg_mins = None
        self._avg_ranges = None

    def add_rescale_point(self, values: List[float]):
        """
        :param values: a list of values to add a point (min, max) for later rescaling
        """
        minimum = numpy.min(values).item() if values else 0.0
        maximum = numpy.max(values).item() if values else 0.0
        self._data.append((minimum, maximum, maximum - minimum))

    def rescale(self, val: float) -> float:
        """
        :param val: the value to rescale
        :return: the rescaled / normalized value based off of previously added points
        """
        if self._avg_mins is None or self._avg_ranges is None:
            self._set_averages()

        rescaled = val - self._avg_mins
        if self._avg_ranges:
            rescaled = rescaled / self._avg_ranges

        return rescaled

    def _set_averages(self):
        self._avg_mins = (
            numpy.average([d[0] for d in self._data]).item() if self._data else 0.0
        )
        self._avg_ranges = (
            numpy.average([d[2] for d in self._data]).item() if self._data else 0.0
        )


class PruningNodeSeriesSmoothingType(Enum):
    """
    Enum for how to smooth a node's pruning estimations / measurements
    """

    none = "none"
    maximum = "maximum"
    minimum = "minimum"


class PruningNodeSeries(object):
    """
    Series of measurements / estimations for a pruning node

    :param measurements: a dictionary containing the measurements for the series
    :param baseline_measurement_key: the baseline key that should be used
        for series comparisons
    :param smoothing_type: the smoothing type to apply to the measurements;
        useful for smoothing out sensitivity measurements for pruning
    :param invert_sensitivity: True to invert the sensitivity values,
        False otherwise
    """

    def __init__(
        self,
        measurements: Union[None, Dict[str, float]],
        baseline_measurement_key: str,
        smoothing_type: PruningNodeSeriesSmoothingType,
        invert_sensitivity: bool,
    ):
        self._baseline = None  # type: Union[None, float]
        self._measurements = []  # type: List[Tuple[float, float]]
        self._measurements_smoothed = []  # type: List[Tuple[float, float]]
        self._smoothing_type = smoothing_type
        self._set_measurements(measurements, baseline_measurement_key)
        self._invert_sensitivity = invert_sensitivity

    @property
    def baseline(self) -> Union[None, float]:
        """
        :return: the baseline measurement value
        """
        return self._baseline

    @property
    def measurements(self) -> List[Tuple[float, float]]:
        """
        :return: the list of measurement tuples (sparsity, measurement)
        """
        return self._measurements

    @property
    def measurements_smoothed(self) -> List[Tuple[float, float]]:
        """
        :return: the list of measurement tuples (sparsity, measurement)
            after applying the smoothing type
        """
        return self._measurements_smoothed

    @property
    def smoothing_type(self) -> PruningNodeSeriesSmoothingType:
        """
        :return: the smoothing type to apply to the measurements;
            useful for smoothing out sensitivity measurements for pruning
        """
        return self._smoothing_type

    @property
    def invert_sensitivity(self) -> bool:
        """
        :return: True to invert the sensitivity values,
            False otherwise
        """
        return self._invert_sensitivity

    def sparse(
        self, sparsity: Union[None, float], smooth: bool = False
    ) -> Union[None, float]:
        """
        :param sparsity: the sparsity to get a measurement for
        :param smooth: True to pull from the measurements_smoothed,
            False otherwise
        :return: the measurement at the given sparsity
        """
        if not self._measurements:
            return None

        if not sparsity:
            return self.baseline

        _, interpolated = interpolate_list_linear(
            self._measurements if not smooth else self._measurements_smoothed, sparsity
        )[0]

        return interpolated

    def sparse_measurements(
        self, smooth: bool = False
    ) -> List[Tuple[float, Union[None, float]]]:
        """
        :param smooth: True to pull from the measurements_smoothed,
            False otherwise
        :return: a list of tuples containing the sparsity from
            0 to 99% at increments of 1% and the associated measurements
        """
        sparsities = [v / 100.0 for v in range(100)]

        if not self._measurements:
            return [v for v in zip(sparsities, [None for _ in range(len(sparsities))])]

        interpolated = interpolate_list_linear(
            self._measurements if not smooth else self._measurements_smoothed,
            sparsities,
        )

        return interpolated

    def sparse_gain(
        self, sparsity: Union[None, float], smooth: bool = False
    ) -> Union[None, float]:
        """
        :param sparsity: the sparsity to get the gain value for
        :param smooth: True to pull from the measurements_smoothed,
            False otherwise
        :return: the ratio of the predicted value at the given sparsity
            as compared with the baseline value
        """
        if not self._measurements:
            return None

        if not sparsity:
            return 1.0

        sparse = self.sparse(sparsity, smooth)

        if not sparse or not self._baseline:
            return 0.0

        return self._baseline / sparse

    def sparse_sensitivity(
        self, sparsity: Union[None, float], smooth: bool = False
    ) -> Union[None, float]:
        """
        :param sparsity: the sparsity to get the sensitivity value for
        :param smooth: True to pull from the measurements_smoothed,
            False otherwise
        :return: the sensitivity comparison (difference) of the measurement
            at the given sparsity compared with the baseline
        """
        sparse = self.sparse(sparsity, smooth)
        baseline = self.baseline

        return PruningNodeSeries._sensitivity(sparse, baseline, self.invert_sensitivity)

    def sparse_desensitivity(
        self, sparsity: Union[None, float], smooth: bool = False
    ) -> Union[None, float]:
        """
        :param sparsity: the sparsity to get the desensitivity value for
        :param smooth: True to pull from the measurements_smoothed,
            False otherwise
        :return: the desensitivity comparison (-1 * difference) of the measurement
            at the given sparsity compared with the baseline
        """
        sparse = self.sparse(sparsity, smooth)
        baseline = self.baseline

        return PruningNodeSeries._sensitivity(
            sparse, baseline, not self.invert_sensitivity
        )

    def sparse_sensitivities(
        self, smooth: bool = False
    ) -> List[Tuple[float, Union[None, float]]]:
        """
        :param smooth: True to pull from the measurements_smoothed,
            False otherwise
        :return: a list of tuples containing the sparsity from
            0 to 99% at increments of 1% and the associated sensitivity value
        """
        measurements = self.sparse_measurements(smooth)
        baseline = self.baseline

        return PruningNodeSeries._sensitivities(
            measurements, baseline, self.invert_sensitivity
        )

    def sparse_desensitivies(
        self, smooth: bool = False
    ) -> List[Tuple[float, Union[None, float]]]:
        """
        :param smooth: True to pull from the measurements_smoothed,
            False otherwise
        :return: a list of tuples containing the sparsity from
            0 to 99% at increments of 1% and the associated desensitivity value
        """
        measurements = self.sparse_measurements(smooth)
        baseline = self.baseline

        return PruningNodeSeries._sensitivities(
            measurements, baseline, not self.invert_sensitivity
        )

    def _set_measurements(
        self, measurements: Dict[str, float], baseline_measurement_key: str,
    ):
        if not measurements:
            return

        meas_min = None
        meas_max = None

        for key, meas in measurements.items():
            meas_smoothed = meas

            if key == baseline_measurement_key:
                self._baseline = meas
            else:
                if meas_min is None or meas < meas_min:
                    meas_min = meas

                if meas_max is None or meas < meas_max:
                    meas_max = meas

                if (
                    self._smoothing_type == PruningNodeSeriesSmoothingType.minimum
                    and meas > meas_min
                ):
                    meas_smoothed = meas_min

                if (
                    self._smoothing_type == PruningNodeSeriesSmoothingType.maximum
                    and meas < meas_max
                ):
                    meas_smoothed = meas_max

            self._measurements.append((float(key), meas))
            self._measurements_smoothed.append((float(key), meas_smoothed))

        self._measurements.sort(key=lambda x: x[0])
        self._measurements_smoothed.sort(key=lambda x: x[0])

    @staticmethod
    def _sensitivity(
        sparse: Union[float, None], baseline: Union[float, None], invert: bool
    ) -> Union[float, None]:
        if sparse is None or baseline is None:
            return None

        sensitivity = (baseline - sparse) if invert else (sparse - baseline)

        return sensitivity

    @staticmethod
    def _sensitivities(
        measurements: List[Tuple[float, Union[float, None]]],
        baseline: Union[float, None],
        invert: bool,
    ):
        sensitivities = []

        for (sparsity, measurement) in measurements:
            sensitivities.append(
                (
                    sparsity,
                    PruningNodeSeries._sensitivity(measurement, baseline, invert),
                )
            )

        return sensitivities


class PruningNodeEvaluator(object):
    """
    Evaluator for a model's node for pruning.
    Able to estimate the effect of pruning on the node for performance, loss, etc

    :param node_id: id of the node to create the evaluator for
    :param model_analysis: analysis of the model
    :param perf_analysis: performance analysis of the model, if any
    :param loss_analysis: loss analysis of the model, if any
    """

    def __init__(
        self,
        node_id: str,
        model_analysis: Dict,
        perf_analysis: Union[None, Dict],
        loss_analysis: Union[None, Dict],
    ):
        self._node_id = node_id
        self._analysis = PruningNodeEvaluator._extract_node_analysis(
            node_id, model_analysis
        )
        self._perf_analysis = PruningNodeEvaluator._extract_node_perf_analysis(
            node_id, perf_analysis
        )
        self._loss_analysis = PruningNodeEvaluator._extract_node_loss_analysis(
            node_id, loss_analysis
        )

        self._params = PruningNodeSeries(
            measurements=OrderedDict(
                [
                    ("0.0", self._analysis["params"]),
                    (
                        "1.0",
                        self._analysis["params"] - self._analysis["prunable_params"],
                    ),
                ]
            ),
            baseline_measurement_key="0.0",
            smoothing_type=PruningNodeSeriesSmoothingType.none,
            invert_sensitivity=False,
        )
        self._flops = PruningNodeSeries(
            measurements=OrderedDict([("0.0", self._analysis["flops"]), ("1.0", 0.0)])
            if self._analysis["flops"]
            else None,
            baseline_measurement_key="0.0",
            smoothing_type=PruningNodeSeriesSmoothingType.none,
            invert_sensitivity=True,
        )
        self._performance = PruningNodeSeries(
            measurements=self._perf_analysis["measurements"]
            if self._perf_analysis
            else None,
            baseline_measurement_key=self._perf_analysis["baseline_measurement_key"]
            if self._perf_analysis
            else None,
            smoothing_type=PruningNodeSeriesSmoothingType.minimum,
            invert_sensitivity=True,
        )
        self._loss = PruningNodeSeries(
            measurements=self._loss_analysis["measurements"]
            if self._loss_analysis
            else None,
            baseline_measurement_key=self._loss_analysis["baseline_measurement_key"]
            if self._loss_analysis
            else None,
            smoothing_type=PruningNodeSeriesSmoothingType.maximum,
            invert_sensitivity=False,
        )
        self._loss_estimated = PruningNodeSeries(
            OrderedDict(
                [
                    ("0.0", 0.0),
                    ("1.0", self._analysis["prunable_equation_sensitivity"]),
                ]
            ),
            baseline_measurement_key="0.0",
            smoothing_type=PruningNodeSeriesSmoothingType.none,
            invert_sensitivity=False,
        )

    @property
    def node_id(self) -> str:
        """
        :return: id of the node the evaluator is created for
        """
        return self._node_id

    @property
    def prunable_params(self) -> Union[int, None]:
        """
        :return: number of prunable params in the node
        """
        return self._analysis["prunable_params"]

    @property
    def params(self) -> PruningNodeSeries:
        """
        :return: the params pruning series for the node
        """
        return self._params

    @property
    def flops(self) -> PruningNodeSeries:
        """
        :return: the flops pruning series for the node
        """
        return self._flops

    @property
    def performance(self) -> PruningNodeSeries:
        """
        :return: the performance timings pruning series for the node
        """
        return self._performance

    @property
    def performance_metric(self) -> PruningNodeSeries:
        """
        :return: the available performance metric,
            falls back on flops if perf sensitivity is not available
        """
        return self.performance if self._perf_analysis is not None else self.flops

    @property
    def loss(self) -> PruningNodeSeries:
        """
        :return: the loss measurements pruning series for the node
        """
        return self._loss

    @property
    def loss_estimated(self) -> PruningNodeSeries:
        """
        :return: the estimated loss measurements pruning series for the node
        """
        return self._loss_estimated

    @property
    def loss_metric(self) -> PruningNodeSeries:
        """
        :return: the available loss metric,
            falls back on estimated loss if loss sensitivity is not available
        """
        return self.loss if self._loss_analysis is not None else self.loss_estimated

    @property
    def structurally_pruned(self) -> bool:
        """
        :return: True if the node is structurally pruned (group convolutions),
            False otherwise
        """
        attributes = (
            self._analysis["attributes"] if "attributes" in self._analysis else None
        )

        return attributes and "group" in attributes and attributes["group"] > 1

    def recoverability(
        self, sparsity: Union[float, None], baseline_sparsity: Union[float, None],
    ) -> Union[float, None]:
        """
        :param sparsity: the sparsity to get recoverability for
        :param baseline_sparsity: the baseline sparsity to use for recoverability
        :return: the estimated confidence of recoverability for the given sparsity
            as compared to the baseline
        """
        baseline = self.loss_metric.sparse_sensitivity(baseline_sparsity)
        estimated = self.loss_metric.sparse_sensitivity(sparsity)

        if baseline is None or estimated is None:
            return None

        if not estimated:
            return 0.0

        return baseline / estimated

    def sparse_costs(
        self,
        balance_perf_loss: float,
        perf_rescaler: ValueRescaler,
        loss_rescaler: ValueRescaler,
    ) -> List[Tuple[str, float, Union[float, None]]]:
        """
        :param balance_perf_loss: the weight [0.0, 1.0] for balancing perf vs loss;
            0.0 for all performance, 1.0 for all loss
        :param perf_rescaler: rescaler to use to rescale vales for performance
            before calculating cost
        :param loss_rescaler: rescaler to use to rescale vales for loss
            before calculating cost
        :return: a list of tuples containing the sparsities from 0% to 99% and
            their associated cost for pruning the node to that sparsity
        """
        perfs = self.performance_metric.sparse_sensitivities(True)
        losses = self.loss_metric.sparse_sensitivities(True)
        costs = []

        for ((sparsity, perf), (_, loss)) in zip(
            perfs, losses
        ):
            perf = (
                perf_rescaler.rescale(perf)
                if perf is not None and perf_rescaler
                else perf
            )
            loss = (
                loss_rescaler.rescale(loss)
                if loss is not None and loss_rescaler
                else loss
            )

            if balance_perf_loss <= 0.0:
                # all performance
                cost = perf
            elif balance_perf_loss >= 1.0:
                # all loss
                cost = loss
            else:
                cost = (
                    balance_perf_loss * loss + (1.0 - balance_perf_loss) * perf
                    if loss is not None and perf is not None
                    else None
                )

            costs.append((self.node_id, sparsity, cost))

        return costs

    @staticmethod
    def _extract_node_analysis(node_id: str, model_analysis: Dict) -> Dict:
        analysis = None

        for node in model_analysis["nodes"]:
            if node["id"] == node_id:
                analysis = node
                break
        assert analysis

        return analysis

    @staticmethod
    def _extract_node_perf_analysis(
        node_id: str, perf_analysis: Union[None, Dict]
    ) -> Union[None, bool, Dict]:
        if not perf_analysis:
            return None

        analysis = False
        for op in perf_analysis["pruning"]["ops"]:
            if op["id"] == node_id:
                analysis = op

        return analysis

    @staticmethod
    def _extract_node_loss_analysis(
        node_id: str, loss_analysis: Union[None, Dict]
    ) -> Union[None, bool, Dict]:
        if not loss_analysis:
            return None

        analysis = False
        for op in loss_analysis["pruning"]["ops"]:
            if op["id"] == node_id:
                analysis = op

        return analysis


class PruningModelEvaluator(object):
    """
    Evaluator for a model for pruning.
    Able to estimate the effect of pruning on a model and each prunable node in a model
    for performance, loss, etc

    :param model_analysis: analysis of the model
    :param perf_analysis: performance analysis of the model, if any
    :param loss_analysis: loss analysis of the model, if any
    """

    MAX_NODE_SPARSITY = 0.975
    EVAL_SENSITIVITY_SPARSITY = 0.95

    def __init__(
        self,
        model_analysis: Dict,
        perf_analysis: Union[None, Dict],
        loss_analysis: Union[None, Dict],
    ):
        self._baseline_time = (
            perf_analysis["baseline"]["model"]["measurement"] if perf_analysis else None
        )
        self._baseline_pruning_time = (
            perf_analysis["pruning"]["model"]["measurements"][
                perf_analysis["pruning"]["model"]["baseline_measurement_key"]
            ]
            if perf_analysis and perf_analysis["pruning"]
            else None
        )

        self._nodes = []  # type: List[PruningNodeEvaluator]
        self._baseline_sparsities = {}  # type: Dict[str, float]
        self._optimized_sparsities = {}  # type: Dict[str, float]
        self._perf_rescaler = None
        self._loss_rescaler = None

        for node in model_analysis["nodes"]:
            if not node["prunable"]:
                continue

            self._nodes.append(
                PruningNodeEvaluator(
                    node["id"], model_analysis, perf_analysis, loss_analysis
                )
            )

    def create_rescale_functions(self):
        """
        Create the loss and performance rescale functions so that later calls
        will be more balanced for optimizing between loss and performance
        """
        self._perf_rescaler = ValueRescaler()
        self._loss_rescaler = ValueRescaler()

        for node in self._nodes:
            if node.performance_metric.measurements:
                self._perf_rescaler.add_rescale_point(
                    [
                        node.performance_metric.sparse_desensitivity(0.0, True),
                        node.performance_metric.sparse_desensitivity(0.95, True),
                    ]
                )

            if node.loss_metric.measurements:
                self._loss_rescaler.add_rescale_point(
                    [
                        node.loss_metric.sparse_sensitivity(0.0, True),
                        node.loss_metric.sparse_sensitivity(0.95, True),
                    ]
                )

    def eval_baseline(self, baseline_sparsity: float):
        """
        Evaluate the baseline (no performance data, only loss) recommended sparsities
        to assign for each node to best maximize recovery.

        :param baseline_sparsity: the baseline_sparsity to use and evaluate with
        """
        self._baseline_sparsities = PruningModelEvaluator._optimize_nodes_sparsity(
            self._nodes,
            baseline_sparsity,
            balance_perf_loss=1.0,
            perf_rescaler=self._perf_rescaler,
            loss_rescaler=self._loss_rescaler,
        )

    def eval_pruning(self, settings: PruningSettings):
        """
        Evaluate the model to assign the evaluate sparsity levels for each node
        in the model given the input pruning settings.

        :param settings: the pruning settings to use and evaluate with
        """
        self._optimized_sparsities = PruningModelEvaluator._optimize_nodes_sparsity(
            self._nodes,
            settings.sparsity,
            balance_perf_loss=settings.balance_perf_loss,
            perf_rescaler=self._perf_rescaler,
            loss_rescaler=self._loss_rescaler,
        )

        for node in self._nodes:
            sparsity = self._optimized_sparsities[node.node_id]
            est_perf_gain = node.performance_metric.sparse_gain(sparsity)
            est_loss = node.loss_metric.sparse(sparsity)

            if sparsity is None or (
                (
                    settings.filter_min_sparsity
                    and sparsity < settings.filter_min_sparsity
                )
                or (
                    settings.filter_min_perf_gain
                    and est_perf_gain is not None
                    and est_perf_gain < settings.filter_min_perf_gain
                )
                or (
                    settings.filter_max_loss_drop
                    and est_loss is not None
                    and est_loss > settings.filter_max_loss_drop
                )
            ):
                self._optimized_sparsities[node.node_id] = None

    def apply_node_overrides(self, node_overrides: List[Dict[str, Any]]):
        """
        Apply any node override sparsity levels to the current evaluated nodes.
        Must be called after eval_pruning if eval_pruning is invoked at all
        to have any effect.

        :param node_overrides: the override sparsity levels for nodes to set with
        """
        for override in node_overrides:
            self._optimized_sparsities[override["node_id"]] = override["sparsity"]

    def to_dict_values(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Create the dictionary values containing the recommended sparsity levels
        for pruning and their estimated times.
        eval_baseline and (eval_pruning and/or apply_node_overrides)
        must be called before

        :return: a tuple containing (model info, list of node info)
        """
        node_values = []

        for node in self._nodes:
            sparsity = self._optimized_sparsities[node.node_id]
            baseline_sparsity = self._baseline_sparsities[node.node_id]
            node_values.append(
                {
                    "node_id": node.node_id,
                    "sparsity": sparsity,
                    "est_recovery": node.recoverability(sparsity, baseline_sparsity),
                    "est_perf_gain": node.performance_metric.sparse_gain(sparsity),
                    "est_time": node.performance.sparse(sparsity),
                    "est_time_baseline": node.performance.baseline,
                    "est_loss_sensitivity": node.loss_metric.sparse_sensitivity(
                        PruningModelEvaluator.EVAL_SENSITIVITY_SPARSITY,
                    ),
                    "est_perf_sensitivity": node.performance_metric.sparse_sensitivity(
                        PruningModelEvaluator.EVAL_SENSITIVITY_SPARSITY,
                    ),
                    "params_baseline": node.params.baseline,
                    "flops_baseline": node.flops.baseline,
                    "params": node.params.sparse(sparsity),
                    "flops": node.flops.sparse(sparsity),
                }
            )

        recoveries = [
            node["est_recovery"]
            for node in node_values
            if node["est_recovery"] is not None
        ]
        loss_sensitivities = [
            node["est_loss_sensitivity"]
            for node in node_values
            if node["est_loss_sensitivity"] is not None
        ]
        perf_sensitivities = [
            node["est_perf_sensitivity"]
            for node in node_values
            if node["est_perf_sensitivity"] is not None
        ]
        est_time_deltas = [
            node["est_time_baseline"] - node["est_time"]
            for node in node_values
            if node["est_time_baseline"] is not None and node["est_time"] is not None
        ]
        params_baseline = [
            node["params_baseline"]
            for node in node_values
            if node["params_baseline"] is not None
        ]
        flops_baseline = [
            node["flops_baseline"]
            for node in node_values
            if node["flops_baseline"] is not None
        ]
        params = [node["params"] for node in node_values if node["params"] is not None]
        flops = [node["flops"] for node in node_values if node["flops"] is not None]

        if est_time_deltas and self._baseline_time and self._baseline_pruning_time:
            est_pruning_time = (
                self._baseline_pruning_time - numpy.sum(est_time_deltas).item()
            )
            est_time = self._baseline_time * (
                est_pruning_time / self._baseline_pruning_time
            )
            est_perf_gain = self._baseline_time / est_time
        else:
            est_time = None
            est_perf_gain = None

        model_values = {
            "est_recovery": numpy.average(recoveries).item() if recoveries else None,
            "est_loss_sensitivity": (
                numpy.average(loss_sensitivities).item() if loss_sensitivities else None
            ),
            "est_perf_sensitivity": (
                numpy.average(perf_sensitivities).item() if perf_sensitivities else None
            ),
            "est_perf_gain": est_perf_gain,
            "est_time": est_time,
            "est_time_baseline": self._baseline_time,
            "params_baseline": sum(params_baseline),
            "flops_baseline": sum(flops_baseline),
            "params": sum(params),
            "flops": sum(flops),
        }

        return node_values, model_values

    @staticmethod
    def _optimize_nodes_sparsity(
        nodes: List[PruningNodeEvaluator],
        sparsity: float,
        balance_perf_loss: float,
        perf_rescaler: ValueRescaler,
        loss_rescaler: ValueRescaler,
    ) -> Dict[str, float]:
        sparsities = {}
        nodes_costs = {}
        costs = []

        for node in nodes:
            sparsities[node.node_id] = None

            if node.structurally_pruned:
                continue

            costs = node.sparse_costs(balance_perf_loss, perf_rescaler, loss_rescaler)

            if costs and costs[0][2] is not None:
                nodes_costs[node.node_id] = costs

        if not nodes_costs:
            return sparsities

        nodes_costs_indices = {node_id: 0 for node_id in nodes_costs.keys()}
        available_steps = len(nodes_costs) * len(costs)
        num_optim_steps = round(available_steps * sparsity)

        for step in range(num_optim_steps):
            smallest_id = None
            smallest_cost = None

            for node_id, cost_index in nodes_costs_indices.items():
                _, cost_sparsity, cost = nodes_costs[node_id][cost_index + 1]

                if cost_sparsity < PruningModelEvaluator.MAX_NODE_SPARSITY and (
                    smallest_cost is None or cost < smallest_cost
                ):
                    smallest_id = node_id
                    smallest_cost = cost

            if smallest_id is None:
                break

            nodes_costs_indices[smallest_id] += 1

        for node_id, cost_index in nodes_costs_indices.items():
            sparsities[node_id] = nodes_costs[node_id][cost_index][1]

        return sparsities
