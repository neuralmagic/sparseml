"""
Helper functions and classes for flask blueprints specific to project optimizations
for pruning
"""


from typing import Callable, Union, NamedTuple, Tuple, Dict, List, Any
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
        self._node_analysis = None
        for node in model_analysis["nodes"]:
            if node["id"] == node_id:
                self._node_analysis = node
                break
        assert self._node_analysis
        self._model_perf_set = perf_analysis and perf_analysis["pruning"]
        self._model_loss_set = loss_analysis and loss_analysis["pruning"]
        self._node_perf = None
        self._node_loss = None

        if self._model_perf_set:
            for op in perf_analysis["pruning"]["ops"]:
                if op["id"] == node_id:
                    self._node_perf = op
                    break

        if self._model_loss_set:
            for op in loss_analysis["pruning"]["ops"]:
                if op["id"] == node_id:
                    self._node_loss = op
                    break

        self._node_perf_measurements = (
            PruningNodeEvaluator._correct_node_measurements(
                self._node_perf["measurements"],
                self._node_perf["baseline_measurement_key"],
            )
            if self._node_perf
            else None
        )
        self._node_perf_measurements_smooth = (
            PruningNodeEvaluator._correct_node_measurements(
                self._node_perf["measurements"],
                self._node_perf["baseline_measurement_key"],
                smooth_min=True,
            )
            if self._node_perf
            else None
        )
        self._node_loss_measurements = (
            PruningNodeEvaluator._correct_node_measurements(
                self._node_loss["measurements"],
                self._node_loss["baseline_measurement_key"],
            )
            if self._node_loss
            else None
        )
        self._node_loss_measurements_smooth = (
            PruningNodeEvaluator._correct_node_measurements(
                self._node_loss["measurements"],
                self._node_loss["baseline_measurement_key"],
                smooth_max=True,
            )
            if self._node_loss
            else None
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
        return self._node_analysis["prunable_params"]

    @property
    def params_baseline(self) -> Union[int, None]:
        """
        :return: number of params in the node at baseline
        """
        return self._node_analysis["params"]

    @property
    def flops_baseline(self) -> Union[int, None]:
        """
        :return: number of flops in the node at baseline
        """
        return self._node_analysis["flops"]

    def sparse_evaluation(
        self,
        sparsity: Union[None, float],
        balance_perf_loss: float,
        rescale_perf_func: Union[None, Callable[[float], float]],
        rescale_loss_func: Union[None, Callable[[float], float]],
    ):
        """
        Evaluate the effect of setting a specific sparsity on the node

        :param sparsity: the sparsity to evaluate with
        :param balance_perf_loss: the perf loss balance weight;
            0.0 for all perf, 1.0 for all loss
        :param rescale_perf_func: a rescale function to normalize the perf values
        :param rescale_loss_func: a rescale function to normalize the loss values
        :return: the effect of the given sparsity on the node
        """
        if sparsity is None:
            return None

        if balance_perf_loss <= 0.0:
            # all perf
            return self.est_perf_sensitivity(sparsity, rescale_perf_func)

        if balance_perf_loss >= 1.0:
            # all loss
            return self.est_loss_sensitivity(sparsity, rescale_loss_func)

        perf = self.est_perf_sensitivity(sparsity, rescale_perf_func)
        loss = self.est_loss_sensitivity(sparsity, rescale_loss_func)

        if perf is None or loss is None:
            return None

        # subtract perf, we want the most sensitive there
        return balance_perf_loss * loss - (1.0 - balance_perf_loss) * perf

    def est_time(
        self, sparsity: Union[float, None], smooth: bool = False
    ) -> Union[float, None]:
        """
        :param sparsity: the sparsity to evaluate with
        :param smooth: True to use smoothed timing info,
            False to use the measured data as is
        :return: the estimated time for the node at a given sparsity level
        """
        if sparsity is None:
            return None

        perf_measurements = (
            self._node_perf_measurements
            if not smooth
            else self._node_perf_measurements_smooth
        )

        if not perf_measurements:
            return None

        return interpolate_list_linear(perf_measurements, sparsity)

    def est_time_baseline(self) -> Union[float, None]:
        """
        :return: the baseline estimated time for the node
        """
        if not self._node_perf:
            return None

        return self._node_perf["measurements"][
            self._node_perf["baseline_measurement_key"]
        ]

    def est_perf_gain(self, sparsity: Union[float, None]) -> Union[float, None]:
        """
        :param sparsity: the sparsity to evaluate with
        :return: the estimated performance gain from setting at a given sparsity level,
            baseline / sparse time
        """
        if sparsity is None:
            return None

        est_baseline = self.est_time_baseline()
        est_sparse = self.est_time(sparsity)

        if est_baseline is None or est_sparse is None:
            return None

        return est_baseline / est_sparse if est_sparse > 0.0 else 0.0

    def est_perf_sensitivity(
        self,
        sparsity: Union[float, None],
        rescale_func: Union[None, Callable[[float], float]],
    ) -> Union[float, None]:
        """
        :param sparsity: the sparsity to evaluate with
        :param rescale_func: a rescale function to normalize the perf values
        :return: the estimated performance sensitivity at a given sparsity level,
            higher is more sensitive and lower is less sensitive
        """
        if sparsity is None:
            return None

        sensitivity = None

        if not self._model_perf_set and self._node_analysis["flops"]:
            # fall back on flops if no performance info is set
            base_flops = self._node_analysis["flops"]
            sparse_flops = (1.0 - sparsity) * base_flops
            # flip so smaller sparse flops give more sensitivity
            sensitivity = -1.0 * (sparse_flops - base_flops)
        elif self._model_perf_set and self._node_perf:
            est_baseline = self.est_time_baseline()
            est_sparse = self.est_time(sparsity, smooth=True)
            # flip so smaller sparse time decreases give more sensitivity
            sensitivity = (
                -1.0 * (est_sparse - est_baseline)
                if est_baseline is not None and est_sparse is not None
                else None
            )

        return (
            rescale_func(sensitivity)
            if rescale_func is not None and sensitivity is not None
            else sensitivity
        )

    def est_loss(
        self, sparsity: Union[float, None], smooth: bool = False
    ) -> Union[float, None]:
        """
        :param sparsity: the sparsity to evaluate with
        :param smooth: True to use smoothed timing info,
            False to use the measured data as is
        :return: the estimated loss for the node at a given sparsity level
        """
        if sparsity is None:
            return None

        loss_measurements = (
            self._node_loss_measurements
            if not smooth
            else self._node_loss_measurements_smooth
        )

        if not loss_measurements:
            return None

        return interpolate_list_linear(loss_measurements, sparsity)

    def est_loss_baseline(self) -> Union[float, None]:
        """
        :return: the baseline estimated loss for the node
        """
        if not self._node_loss:
            return None

        return self._node_loss["measurements"][
            self._node_loss["baseline_measurement_key"]
        ]

    def est_loss_sensitivity(
        self,
        sparsity: Union[float, None],
        rescale_func: Union[None, Callable[[float], float]],
    ) -> Union[float, None]:
        """
        :param sparsity: the sparsity to evaluate with
        :param rescale_func: a rescale function to normalize the loss values
        :return: the estimated loss sensitivity at a given sparsity level,
            higher is more sensitive and lower is less sensitive
        """
        if sparsity is None:
            return None

        sensitivity = None

        if (
            not self._model_loss_set
            and self.prunable_params
            and self._node_analysis["input_shapes"]
            and self._node_analysis["output_shapes"]
        ):
            input_size = numpy.prod(
                [
                    numpy.prod([s for i, s in enumerate(shape) if s and i > 0])
                    for shape in self._node_analysis["input_shapes"]
                    if shape
                ]
            )
            output_size = numpy.prod(
                [
                    numpy.prod([s for i, s in enumerate(shape) if s and i > 0])
                    for shape in self._node_analysis["output_shapes"]
                    if shape
                ]
            )
            node_volume = input_size + output_size
            baseline_params = self.prunable_params
            sparse_params = (1.0 - sparsity) * baseline_params
            baseline = node_volume / baseline_params if baseline_params > 0 else 0.0
            sparse = node_volume / sparse_params if sparse_params > 0 else 0.0
            sensitivity = sparse - baseline
        elif self._model_loss_set and self._node_loss:
            est_baseline = self.est_loss_baseline()
            est_sparse = self.est_loss(sparsity, smooth=True)
            sensitivity = (
                (est_sparse - est_baseline)
                if est_baseline is not None and est_sparse is not None
                else None
            )

        return (
            rescale_func(sensitivity)
            if rescale_func is not None and sensitivity is not None
            else sensitivity
        )

    def est_recoverability(
        self,
        sparsity: Union[float, None],
        baseline_sparsity: Union[float, None],
        rescale_func: Union[None, Callable[[float], float]],
    ) -> Union[None, float]:
        """
        :param sparsity: the sparsity to evaluate with
        :param baseline_sparsity: the baseline sparsity to compare against
        :param rescale_func: a rescale function to normalize the loss values
        :return: the estimated likelihood of recovery at a given sparsity level,
            baseline_loss_sensitivity / sparse_loss_sensitivity
        """
        if sparsity is None:
            return None

        if baseline_sparsity is None:
            return None

        baseline = self.est_loss_sensitivity(baseline_sparsity, rescale_func)
        estimated = self.est_loss_sensitivity(sparsity, rescale_func)

        if baseline is None or estimated is None:
            return None

        return baseline / (estimated if estimated > 0.0 else 0.001)

    @staticmethod
    def _correct_node_measurements(
        orig_measurements: Dict[str, float],
        baseline_measurement_key: str,
        smooth_max: bool = False,
        smooth_min: bool = False,
    ) -> List[Tuple[float, float]]:
        measurements = []
        min_sparse_meas = None
        max_sparse_meas = None

        for key, meas in orig_measurements.items():
            if (
                min_sparse_meas is None
                or meas < min_sparse_meas
                and key != baseline_measurement_key
            ):
                min_sparse_meas = meas

            if (
                max_sparse_meas is None
                or meas < max_sparse_meas
                and key != baseline_measurement_key
            ):
                max_sparse_meas = meas

            if smooth_min and meas > min_sparse_meas:
                meas = min_sparse_meas

            if smooth_max and meas < max_sparse_meas:
                meas = max_sparse_meas

            measurements.append((float(key), meas))

        measurements.sort(key=lambda x: x[0])

        return measurements


class _PruningNodeEvalWrapper(object):
    def __init__(self, node: PruningNodeEvaluator):
        self.node = node
        self.baseline_sparsity = None
        self.optimized_sparsity = None

    @property
    def optimized_params(self) -> int:
        """
        :return: the params at optimized sparsity if optimized sparsity is not none,
            otherwise return baseline params
        """
        if self.optimized_sparsity:
            return (self.node.prunable_params) * (1 - self.optimized_sparsity) + (
                self.node.params_baseline - self.node.prunable_params
            )
        else:
            return self.node.params_baseline

    @property
    def optimized_flops(self) -> int:
        """
        :return: the flops at optimized sparsity if optimized sparsity is not none,
            otherwise return baseline params
        """
        if self.optimized_sparsity:
            return self.node.flops_baseline * (1 - self.optimized_sparsity)
        else:
            return self.node.flops_baseline


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

        self._nodes = []  # type: List[_PruningNodeEvalWrapper]
        self._rescale_perf_func = None
        self._rescale_loss_func = None
        self._baseline_sparsities = []
        self._optimized_sparsities = []

        for node in model_analysis["nodes"]:
            if not node["prunable"]:
                continue

            self._nodes.append(
                _PruningNodeEvalWrapper(
                    PruningNodeEvaluator(
                        node["id"], model_analysis, perf_analysis, loss_analysis
                    )
                )
            )

    def create_rescale_functions(self):
        """
        Create the loss and performance rescale functions so that later calls
        will be more balanced for optimizing between loss and performance
        """

        def _create_rescale_func(
            mins: List[float], ranges: List[float]
        ) -> Callable[[float], float]:
            mins_avg = numpy.average(mins).item() if mins else 0.0
            ranges_avg = numpy.average(ranges).item() if ranges else 1.0

            # shift so mins are at 0 on average and ranges + mins are at 1 on average
            return lambda val: (val - mins_avg) / ranges_avg

        perf_mins = []
        perf_ranges = []
        loss_mins = []
        loss_ranges = []

        for node in self._nodes:
            perf_min = node.node.est_perf_sensitivity(0.0, None)
            perf_max = node.node.est_perf_sensitivity(0.95, None)  # 95% ignore extrema
            if perf_min is not None:
                perf_mins.append(perf_min)
                if perf_max is not None:
                    perf_ranges.append(perf_max - perf_min)

            loss_min = node.node.est_loss_sensitivity(0.0, None)
            loss_max = node.node.est_loss_sensitivity(0.95, None)  # 95% ignore extrema
            if loss_min is not None:
                loss_mins.append(loss_min)
                if loss_max is not None:
                    loss_ranges.append(loss_max - loss_min)

        self._rescale_perf_func = _create_rescale_func(perf_mins, perf_ranges)
        self._rescale_loss_func = _create_rescale_func(loss_mins, loss_ranges)

    def eval_baseline(self, baseline_sparsity: float):
        """
        Evaluate the baseline (no performance data, only loss) recommended sparsities
        to assign for each node to best maximize recovery.

        :param baseline_sparsity: the baseline_sparsity to use and evaluate with
        """
        sparsities = PruningModelEvaluator._optimize_nodes_sparsity(
            [node.node for node in self._nodes],
            baseline_sparsity,
            balance_perf_loss=1.0,
            rescale_perf_func=self._rescale_perf_func,
            rescale_loss_func=self._rescale_loss_func,
        )

        for node, sparsity in zip(self._nodes, sparsities):
            node.baseline_sparsity = sparsity

    def eval_pruning(self, settings: PruningSettings):
        """
        Evaluate the model to assign the evaluate sparsity levels for each node
        in the model given the input pruning settings.

        :param settings: the pruning settings to use and evaluate with
        """
        sparsities = PruningModelEvaluator._optimize_nodes_sparsity(
            [node.node for node in self._nodes],
            settings.sparsity,
            balance_perf_loss=settings.balance_perf_loss,
            rescale_perf_func=self._rescale_perf_func,
            rescale_loss_func=self._rescale_loss_func,
        )

        for node, sparsity in zip(self._nodes, sparsities):
            est_perf_gain = node.node.est_perf_gain(sparsity)
            est_loss = node.node.est_loss(sparsity)

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
                node.optimized_sparsity = None
            else:
                node.optimized_sparsity = sparsity

    def apply_node_overrides(self, node_overrides: List[Dict[str, Any]]):
        """
        Apply any node override sparsity levels to the current evaluated nodes.
        Must be called after eval_pruning if eval_pruning is invoked at all
        to have any effect.

        :param node_overrides: the override sparsity levels for nodes to set with
        """
        for override in node_overrides:
            for node in self._nodes:
                if override["node_id"] == node.node.node_id:
                    node.optimized_sparsity = override["sparsity"]
                    break

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
            node_values.append(
                {
                    "node_id": node.node.node_id,
                    "sparsity": node.optimized_sparsity,
                    "est_recovery": node.node.est_recoverability(
                        node.optimized_sparsity,
                        node.baseline_sparsity,
                        self._rescale_loss_func,
                    ),
                    "est_perf_gain": node.node.est_perf_gain(node.optimized_sparsity),
                    "est_time": node.node.est_time(node.optimized_sparsity),
                    "est_time_baseline": node.node.est_time_baseline(),
                    "est_loss_sensitivity": node.node.est_loss_sensitivity(
                        PruningModelEvaluator.EVAL_SENSITIVITY_SPARSITY,
                        self._rescale_loss_func,
                    ),
                    "est_perf_sensitivity": node.node.est_perf_sensitivity(
                        PruningModelEvaluator.EVAL_SENSITIVITY_SPARSITY,
                        self._rescale_perf_func,
                    ),
                    "params_baseline": node.node.params_baseline,
                    "flops_baseline": node.node.flops_baseline,
                    "params": node.optimized_params,
                    "flops": node.optimized_flops,
                }
            )

        recoveries = [
            node["est_recovery"]
            for node in node_values
            if node["est_recovery"] is not None
        ]
        est_time_deltas = [
            node["est_time_baseline"] - node["est_time"]
            for node in node_values
            if node["est_time_baseline"] is not None and node["est_time"] is not None
        ]

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
            "est_perf_gain": est_perf_gain,
            "est_time": est_time,
            "est_time_baseline": self._baseline_time,
            "params_baseline": 0,
            "flops_baseline": 0,
            "params": 0,
            "flops": 0,
        }

        for node in self._nodes:
            model_values["params_baseline"] += node.node.params_baseline
            model_values["flops_baseline"] += node.node.params_baseline
            model_values["params"] += node.optimized_params
            model_values["flops"] += node.optimized_flops

        return node_values, model_values

    @staticmethod
    def _optimize_nodes_sparsity(
        nodes: List[PruningNodeEvaluator],
        sparsity: float,
        balance_perf_loss: float,
        rescale_perf_func: Union[None, Callable[[float], float]],
        rescale_loss_func: Union[None, Callable[[float], float]],
    ):
        sparsities = [
            0.0
            if node.prunable_params
            and node.sparse_evaluation(
                sparsity, balance_perf_loss, rescale_perf_func, rescale_loss_func
            )
            else None
            for node in nodes
        ]
        total_sparsity = PruningModelEvaluator._eval_total_sparsity(nodes, sparsities)

        while total_sparsity is not None and total_sparsity < sparsity:
            PruningModelEvaluator._apply_cheapest_node_step(
                nodes,
                sparsities,
                balance_perf_loss,
                rescale_perf_func,
                rescale_loss_func,
                step_size=0.01,
            )
            total_sparsity = PruningModelEvaluator._eval_total_sparsity(
                nodes, sparsities
            )

        return sparsities

    @staticmethod
    def _eval_total_sparsity(
        nodes: List[PruningNodeEvaluator], sparsities: List[Union[float, None]]
    ) -> Union[None, float]:
        current = 0.0
        total = 0.0

        for node, sparsity in zip(nodes, sparsities):
            if sparsity is None:
                continue

            total += node.prunable_params
            current += sparsity * node.prunable_params

        return current / total if total > 0.0 else None

    @staticmethod
    def _apply_cheapest_node_step(
        nodes: List[PruningNodeEvaluator],
        sparsities: List[Union[float, None]],
        balance_perf_loss: float,
        rescale_perf_func: Union[None, Callable[[float], float]],
        rescale_loss_func: Union[None, Callable[[float], float]],
        step_size: float,
    ):
        sparsity_costs = [None for _ in sparsities]  # type: List[Union[None, float]]

        for index, (node, sparsity) in enumerate(zip(nodes, sparsities)):
            if sparsity is None or sparsity >= PruningModelEvaluator.MAX_NODE_SPARSITY:
                continue

            current_eval = node.sparse_evaluation(
                sparsity, balance_perf_loss, rescale_perf_func, rescale_loss_func
            )
            next_eval = node.sparse_evaluation(
                sparsity + step_size,
                balance_perf_loss,
                rescale_perf_func,
                rescale_loss_func,
            )
            cost = next_eval - current_eval
            sparsity_costs[index] = cost

        # zero costs
        valid_costs = [c for c in sparsity_costs if c is not None]
        costs_min = min(valid_costs) if valid_costs else 0.0
        sparsity_costs = [
            c - costs_min if c is not None else None for c in sparsity_costs
        ]

        # normalize and flip costs to adjustments
        # more cost == smaller gradient
        # distribute step size
        costs_sum = sum([c for c in sparsity_costs if c is not None])
        sparsity_adjustments = [
            (1.0 - (c / costs_sum)) * step_size if c is not None else 0.0
            for c in sparsity_costs
        ]

        for index, (sparsity, adjustment) in enumerate(
            zip(sparsities, sparsity_adjustments)
        ):
            sparsities[index] = sparsity + adjustment
