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
Generic code related to sensitivity analysis.
"""

import json
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import numpy
import pandas

import matplotlib.pyplot as plt
from sparseml.utils.helpers import clean_path, create_parent_dirs, interpolated_integral


__all__ = [
    "default_pruning_sparsities_loss",
    "default_pruning_sparsities_perf",
    "PruningSensitivityResult",
    "PruningLossSensitivityAnalysis",
    "PruningPerfSensitivityAnalysis",
    "LRLossSensitivityAnalysis",
]


def default_pruning_sparsities_loss(extended: bool) -> Tuple[float, ...]:
    """
    The default sparsities to use for checking pruning effects on the loss

    :param extended: extend the sparsties to return a full range
        instead of a subset of target sparstiies
    :return: the sparsities to check for effects on the loss
    """
    if not extended:
        return 0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99

    sparsities = [float(s) / 100.0 for s in range(100)]

    return tuple(sparsities)


def default_pruning_sparsities_perf() -> Tuple[float, ...]:
    """
    :return: the sparsities to check for effects on the loss
    """
    return 0.0, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99


class PruningSensitivityResult(object):
    """
    A sensitivity result for a given node/param in a model.
    Ex: loss sensitivity or perf sensitivity

    :param id_: id for the node / param
    :param name: human readable name for the node / param
    :param index: index order for when the node / param is used in the model
    :param baseline_measurement_index: index for where the baseline measurement
        is stored in the sparse_measurements, if any
    :param sparse_measurements: the sparse measurements to prepopulate with, if any
    """

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]):
        """
        Create a new loss sensitivity result from a dictionary of values.
        Expected to match the format as given in the dict() call.

        :param dictionary: the dictionary to create a result out of
        :return: the created KSLossSensitivityResult
        """
        return PruningSensitivityResult(
            dictionary["id"],
            dictionary["name"],
            dictionary["index"],
            dictionary["baseline_measurement_index"],
            dictionary["baseline_measurement_key"],
            OrderedDict(dictionary["sparse_measurements"]),
        )

    def __init__(
        self,
        id_: str,
        name: str,
        index: int,
        baseline_measurement_index: int = -1,
        baseline_measurement_key: str = None,
        sparse_measurements: Dict[float, List[float]] = None,
    ):
        self._id = id_
        self._name = name
        self._index = index
        self._baseline_measurement_index = baseline_measurement_index
        self._baseline_measurement_key = baseline_measurement_key
        self._sparse_measurements = (
            OrderedDict() if sparse_measurements is None else sparse_measurements
        )  # type: Dict[float, List[float]]

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.dict())

    @property
    def id_(self) -> str:
        """
        :return: id for the node / param
        """
        return self._id

    @property
    def name(self) -> str:
        """
        :return: human readable name for the node / param
        """
        return self._name

    @property
    def index(self) -> int:
        """
        :return: index order for when the node / param is used in the model
        """
        return self._index

    @property
    def baseline_measurement_index(self) -> int:
        """
        :return: index for where the baseline measurement
            is stored in the sparse_measurements, if any
        """
        return self._baseline_measurement_index

    @property
    def baseline_measurement_key(self) -> Union[None, float]:
        """
        :return: key for where the baseline measurement
            is stored in the sparse_measurements, if any
        """
        return self._baseline_measurement_key

    @property
    def has_baseline(self) -> bool:
        """
        :return: True if the result has a baseline measurement in the
            sparse_measurements, False otherwise
        """
        return self.baseline_measurement_index > -1

    @property
    def sparse_measurements(self) -> Dict[float, List[float]]:
        """
        :return: the sparse measurements
        """
        return self._sparse_measurements

    @property
    def averages(self) -> Dict[float, float]:
        """
        :return: average values of loss for each level recorded
        """
        averages = OrderedDict()

        for sparsity, values in self.sparse_measurements.items():
            averages[sparsity] = numpy.mean(values).item()

        return averages

    @property
    def baseline_average(self) -> float:
        """
        :return: the baseline average time to compare to for the result
        """
        if not self.has_baseline:
            raise ValueError("baseline is not available in this measurement")

        averages = self.averages

        return averages[list(averages.keys())[self.baseline_measurement_index]]

    @property
    def sparse_average(self) -> float:
        """
        :return: average loss across all levels recorded
        """
        averages = [val for val in self.averages.items()]

        return numpy.mean(averages).item()

    @property
    def sparse_integral(self) -> float:
        """
        :return: integrated loss across all levels recorded
        """
        measurements = [(key, val) for key, val in self.averages.items()]
        integral = interpolated_integral(measurements)

        return integral

    def sparse_comparison(self, compare_index: int = -1):
        """
        Compare the baseline average to a sparse average value through the difference:
        sparse - baseline

        If compare_index is not given then will compare
        with the sparsity closest to 90%.
        90% is used as a reasonable achievable baseline to keep from introducing too
        much noise at the extremes of the tests.

        If not has_baseline, then will compare against the first index.

        :param compare_index: the index to compare against the baseline with,
            if not supplied will compare against the sparsity measurement closest to 90%
        :return: a comparison of the sparse average with the baseline
            (sparse - baseline)
        """
        averages = self.averages

        if len(averages) < 2:
            return 0.0

        baseline_index = self.baseline_measurement_index if self.has_baseline else 0
        baseline = averages[list(averages.keys())[baseline_index]]

        if compare_index < 0:
            sparsity_norms_90 = [
                (abs(sparsity - 0.9), sparsity, avg)
                for sparsity, avg in averages.items()
            ]
            sparsity_norms_90.sort(key=lambda v: v[0])
            compare = sparsity_norms_90[0][2]
        else:
            compare = averages[list(averages.keys())[compare_index]]

        return compare - baseline

    def dict(self) -> Dict[str, Any]:
        """
        :return: dictionary representation of the current instance
        """

        return {
            "id": self.id_,
            "name": self.name,
            "index": self.index,
            "baseline_measurement_index": self.baseline_measurement_index,
            "has_baseline": self.has_baseline,
            "baseline_measurement_key": self.baseline_measurement_key,
            "sparse_measurements": [
                (key, val) for key, val in self.sparse_measurements.items()
            ],
            "averages": self.averages,
            "baseline_average": self.baseline_average if self.has_baseline else None,
            "sparse_average": self.sparse_average,
            "sparse_integral": self.sparse_integral,
            "sparse_comparison": self.sparse_comparison(),
        }

    def add_measurement(self, sparsity: float, loss: float, baseline: bool):
        """
        add a sparse measurement to the result

        :param sparsity: the sparsity the measurement was performed at
        :param loss: resulting loss for the given sparsity for the measurement
        :param baseline: True if this is a baseline measurement, False otherwise
        """
        if sparsity not in self._sparse_measurements:
            if baseline:
                self._baseline_measurement_index = len(self._sparse_measurements)
                self._baseline_measurement_key = sparsity

            self._sparse_measurements[sparsity] = []

        self._sparse_measurements[sparsity].append(loss)


class PruningLossSensitivityAnalysis(object):
    """
    Analysis result for how kernel sparsity (pruning) affects the loss of a given model.
    Contains layer by layer results.
    """

    @staticmethod
    def load_json(path: str):
        """
        :param path: the path to load a previous analysis from
        :return: the KSLossSensitivityAnalysis instance from the json
        """
        path = clean_path(path)

        with open(path, "r") as file:
            objs = json.load(file)

        return PruningLossSensitivityAnalysis.from_dict(objs)

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]):
        """
        :param dictionary: the dictionary to create an analysis object from
        :return: the KSLossSensitivityAnalysis instance from the json
        """
        analysis = PruningLossSensitivityAnalysis()

        for res_obj in dictionary["results"]:
            analysis._results.append(PruningSensitivityResult.from_dict(res_obj))

        return analysis

    def __init__(self):
        self._results = []  # type: List[PruningSensitivityResult]

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.dict())

    @property
    def results_model(self) -> PruningSensitivityResult:
        """
        :return: the overall results for the model
        """
        results_model = PruningSensitivityResult(
            id_="__model__", name="__model__", index=-1
        )
        measurements = {}
        baseline_index = None

        for res in self.results:
            for key, val in res.averages.items():
                if key not in measurements:
                    measurements[key] = 0.0
                measurements[key] += val

            if res.has_baseline:
                baseline_index = res.baseline_measurement_index

        for index, (key, val) in enumerate(measurements.items()):
            results_model.add_measurement(key, val, index == baseline_index)

        return results_model

    @property
    def results(self) -> List[PruningSensitivityResult]:
        """
        :return: the individual results for the analysis
        """
        return self._results

    def dict(self) -> Dict[str, Any]:
        """
        :return: dictionary representation of the current instance
        """
        return {"results": [res.dict() for res in self._results]}

    def add_result(
        self,
        id_: Union[str, None],
        name: str,
        index: int,
        sparsity: float,
        measurement: float,
        baseline: bool,
    ):
        """
        Add a result to the sensitivity analysis for a specific param

        :param id_: the identifier to add the result for
        :param name: the readable name to add the result for
        :param index: the index of the param as found in the model parameters
        :param sparsity: the sparsity to add the result for
        :param measurement: the loss measurement to add the result for
        :param baseline: True if this is a baseline measurement, False otherwise
        """
        result = [
            res
            for res in self._results
            if (res.id_ == id_ and id_ is not None)
            or (res.name == name and id_ is None)
        ]

        if len(result) > 0:
            result = result[0]
        else:
            result = PruningSensitivityResult(id_, name, index)
            self._results.append(result)

        result.add_measurement(sparsity, measurement, baseline)

    def get_result(self, id_or_name: str) -> PruningSensitivityResult:
        """
        get a result from the sensitivity analysis for a specific param

        :param id_or_name: the id or name to get the result for
        :return: the loss sensitivity results for the given id or name
        """
        for res in self._results:
            if id_or_name == res.id_ or id_or_name == res.name:
                return res

        raise ValueError("could not find id_or_name {} in results".format(id_or_name))

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

    def plot(
        self,
        path: Union[str, None],
        plot_integral: bool,
        normalize: bool = True,
        title: str = None,
    ) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[None, None]]:
        """
        :param path: the path to save an img version of the chart,
            None to display the plot
        :param plot_integral: True to plot the calculated loss integrals for each layer,
            False to plot the averages
        :param normalize: normalize the values to a unit distribution (0 mean, 1 std)
        :param title: the title to put on the chart
        :return: the created figure and axes if path is None, otherwise (None, None)
        """
        names = [
            "{} ({})".format(res.name, res.id_) if res.id_ is not None else res.name
            for res in self._results
        ]
        values = [
            res.sparse_integral if plot_integral else res.sparse_average
            for res in self._results
        ]

        if normalize:
            mean = numpy.mean(values)
            std = numpy.std(values)
            values = [(val - mean) / std for val in values]

        height = round(len(names) / 4) + 3
        fig = plt.figure(figsize=(12, height))
        ax = fig.add_subplot(111)

        if title is not None:
            ax.set_title(title)

        ax.invert_yaxis()
        frame = pandas.DataFrame(
            list(zip(names, values)), columns=["Param", "Sensitivity"]
        )
        frame.plot.barh(ax=ax, x="Param", y="Sensitivity")
        plt.gca().invert_yaxis()

        if path is None:
            plt.show()

            return fig, ax

        path = clean_path(path)
        create_parent_dirs(path)
        plt.savefig(path)
        plt.close(fig)

        return None, None

    def print_res(self):
        """
        Print the recorded sensitivity values results
        """

        print("KS Sensitivity")
        print("\tLoss Avg\t\tLoss Int\t\tId\t\tName")

        for res in self._results:
            print(
                "\t{:.4f}\t\t{:.4f}\t\t{}\t\t{}".format(
                    res.sparse_average, res.sparse_integral, res.id_, res.name
                )
            )


class PruningPerfSensitivityAnalysis(object):
    """
    Analysis result for how kernel sparsity (pruning) affects the loss of a given model.
    Contains layer by layer results.

    :param num_cores: number of physical cpu cores the analysis was run on
    :param batch_size: the input batch size the analysis was run for
    """

    @staticmethod
    def load_json(path: str):
        """
        :param path: the path to load a previous analysis from
        :return: the KSPerfSensitivityAnalysis instance from the json
        """
        path = clean_path(path)

        with open(path, "r") as file:
            objs = json.load(file)

        return PruningPerfSensitivityAnalysis.from_dict(objs)

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]):
        """
        :param dictionary: the dictionary to create an analysis object from
        :return: the KSPerfSensitivityAnalysis instance from the json
        """
        analysis = PruningPerfSensitivityAnalysis(
            dictionary["num_cores"], dictionary["batch_size"]
        )
        analysis._results_model = PruningSensitivityResult.from_dict(
            dictionary["results_model"]
        )

        for res_obj in dictionary["results"]:
            analysis._results.append(PruningSensitivityResult.from_dict(res_obj))

        return analysis

    def __init__(self, num_cores: int, batch_size: int):
        self._num_cores = num_cores
        self._batch_size = batch_size
        self._results = []  # type: List[PruningSensitivityResult]
        self._results_model = PruningSensitivityResult("__model__", "model", -1)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.dict())

    @property
    def num_cores(self) -> int:
        """
        :return: number of physical cpu cores the analysis was run on
        """
        return self._num_cores

    @property
    def batch_size(self) -> int:
        """
        :return: the input batch size the analysis was run for
        """
        return self._batch_size

    @property
    def results_model(self) -> PruningSensitivityResult:
        """
        :return: the overall results for the model
        """
        return self._results_model

    @property
    def results(self) -> List[PruningSensitivityResult]:
        """
        :return: the individual results for the analysis
        """
        return self._results

    def dict(self) -> Dict[str, Any]:
        """
        :return: dictionary representation of the current instance
        """
        return {
            "num_cores": self._num_cores,
            "batch_size": self._batch_size,
            "results_model": self._results_model.dict(),
            "results": [res.dict() for res in self._results],
        }

    def add_model_result(self, sparsity: float, measurement: float, baseline: bool):
        """
        Add a result to the sensitivity analysis for the overall model

        :param sparsity: the sparsity to add the result for
        :param measurement: resulting timing in seconds for the given sparsity
            for the measurement
        :param baseline: True if this is a baseline measurement, False otherwise
        """
        self._results_model.add_measurement(sparsity, measurement, baseline)

    def add_result(
        self,
        id_: Union[str, None],
        name: str,
        index: int,
        sparsity: float,
        measurement: float,
        baseline: bool,
    ):
        """
        Add a result to the sensitivity analysis for a specific param

        :param id_: the identifier to add the result for
        :param name: the readable name to add the result for
        :param index: the index of the param as found in the model parameters
        :param sparsity: the sparsity to add the result for
        :param measurement: resulting timing in seconds for the given sparsity
            for the measurement
        :param baseline: True if this is a baseline measurement, False otherwise
        """
        result = [
            res
            for res in self._results
            if (res.id_ == id_ and id_ is not None)
            or (res.name == name and id_ is None)
        ]

        if len(result) > 0:
            result = result[0]
        else:
            result = PruningSensitivityResult(id_, name, index)
            self._results.append(result)

        result.add_measurement(sparsity, measurement, baseline)

    def get_result(self, id_or_name: str) -> PruningSensitivityResult:
        """
        get a result from the sensitivity analysis for a specific param

        :param id_or_name: the id or name to get the result for
        :return: the loss sensitivity results for the given id or name
        """
        for res in self._results:
            if id_or_name == res.id_ or id_or_name == res.name:
                return res

        raise ValueError("could not find id_or_name {} in results".format(id_or_name))

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

    def plot(
        self,
        path: Union[str, None],
        title: str = None,
    ) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[None, None]]:
        """
        :param path: the path to save an img version of the chart,
            None to display the plot
        :param title: the title to put on the chart
        :return: the created figure and axes if path is None, otherwise (None, None)
        """
        names = ["{} ({})".format(res.name, res.id_) for res in self._results]
        values = [res.sparse_comparison() for res in self._results]

        height = round(len(names) / 4) + 3
        fig = plt.figure(figsize=(12, height))
        ax = fig.add_subplot(111)

        if title is not None:
            ax.set_title(title)

        ax.invert_yaxis()
        frame = pandas.DataFrame(
            list(zip(names, values)), columns=["Param", "Sparse comparison (ms)"]
        )
        frame.plot.barh(ax=ax, x="Param", y="Sparse comparison (ms)")
        plt.gca().invert_yaxis()

        if path is None:
            plt.show()

            return fig, ax

        path = clean_path(path)
        create_parent_dirs(path)
        plt.savefig(path)
        plt.close(fig)

        return None, None

    def print_res(self):
        """
        Print the recorded sensitivity values results
        """

        print("KS Sensitivity")
        print("\tPerf Avg\t\tPerf Int\t\tId\t\tName")

        for res in self._results:
            print(
                "\t{:.4f}\t\t{:.4f}\t\t{}\t\t{}".format(
                    res.sparse_average, res.sparse_integral, res.id_, res.name
                )
            )


class LRLossSensitivityAnalysis(object):
    """
    Basic class for tracking the results from a learning rate sensitivity analysis
    """

    @staticmethod
    def load_json(path: str):
        """
        :param path: the path to load a previous analysis from
        :return: the analysis instance created from the json file
        """
        with open(path, "r") as file:
            objs = json.load(file)

        analysis = LRLossSensitivityAnalysis()
        for res_obj in objs["results"]:
            analysis.add_result(res_obj["lr"], res_obj["loss_measurements"])

        return analysis

    def __init__(self):
        self._results = []

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.dict())

    @property
    def results(self) -> List[Dict[str, Any]]:
        return self._results

    def add_result(self, lr: float, loss_measurements: List[float]):
        self._results.append(
            {
                "lr": lr,
                "loss_measurements": loss_measurements,
                "loss_avg": numpy.mean(loss_measurements).item(),
            }
        )

    def dict(self) -> Dict[str, Any]:
        """
        :return: dictionary representation of the current instance
        """
        return {"results": self._results}

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

    def plot(
        self,
        path: Union[str, None],
        title: str = None,
    ) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[None, None]]:
        """
        Plot the recorded sensitivity values

        :param path: the path for where to save the plot,
            if not supplied will display it
        :param title: the title of the plot to apply,
            defaults to '{plot_loss_key} LR Sensitivity'
        :return: the figure and axes if the figure was displayed; else None, None
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        if title is None:
            title = ""
        elif title == "__default__":
            title = "LR Sensitivity"

        ax.set_title(title)
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Avg Loss")
        frame = pandas.DataFrame.from_records(
            [(lr_res["lr"], lr_res["loss_avg"]) for lr_res in self._results],
            columns=["Learning Rate", "Avg Loss"],
        )
        frame.plot(x="Learning Rate", y="Avg Loss", marker=".", logx=True, ax=ax)

        if path is None:
            plt.show()

            return fig, ax

        path = clean_path(path)
        create_parent_dirs(path)
        plt.savefig(path)
        plt.close(fig)

        return None, None

    def print_res(self):
        """
        Print the recorded sensitivity values CSV results
        """

        print("LR Sensitivity")
        print("\tLR\t\tLoss")

        for lr_res in self._results:
            print("\t{:.4E}\t\t{:.4f}".format(lr_res["lr"], lr_res["loss_avg"]))
