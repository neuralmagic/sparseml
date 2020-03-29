"""
generic code related to sensitivity analysis
"""

from typing import Tuple, List, Dict, Any, Union
import json
from tqdm import auto
import numpy
import matplotlib.pyplot as plt
import pandas

from neuralmagicML.utils.helpers import clean_path, create_parent_dirs


__all__ = [
    "KSLossSensitivityResult",
    "KSLossSensitivityAnalysis",
    "KSLossSensitivityProgress",
]


class KSLossSensitivityResult(object):
    """
    Sensitivity results for a module's (layer) param
    """

    def __init__(
        self,
        name: str,
        param_name: str,
        type_: str,
        measured: List[Tuple[float, float]] = None,
    ):
        """
        :param name: name of the module or layer in the parent
        :param param_name: name of the param that was analyzed
        :param type_: type of layer; ex: conv, linear, etc
        :param measured: the measured results, a list of tuples ordered as follows [(sparsity, loss)]
        """
        self.name = name
        self.param_name = param_name
        self.type_ = type_
        self.measured = measured

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.dict())

    @property
    def integral(self) -> float:
        """
        :return: calculate the approximated integral for the sensitivity using the measured results
                 returns the approximated area under the sparsity vs loss curve
        """
        total = 0.0
        total_dist = 0.0

        for index, (sparsity, loss) in enumerate(self.measured):
            prev_distance = (
                sparsity
                if index == 0
                else (sparsity - self.measured[index - 1][0]) / 2.0
            )
            next_distance = (
                1.0 - sparsity
                if index == len(self.measured) - 1
                else (self.measured[index + 1][0] - sparsity) / 2.0
            )
            x_dist = prev_distance + next_distance
            total_dist += x_dist
            total += x_dist * loss

        return total

    def dict(self) -> Dict[str, Any]:
        """
        :return: dictionary representation of the current instance
        """
        return {
            "name": self.name,
            "param_name": self.param_name,
            "type_": self.type_,
            "measured": [{"sparsity": val[0], "loss": val[1]} for val in self.measured],
            "integral_loss": self.integral,
        }


class KSLossSensitivityAnalysis(object):
    @staticmethod
    def load_json(path: str):
        """
        :param path: the path to load a previous analysis from
        :return: the analysis instance created from the json file
        """
        with open(path, "r") as file:
            objs = json.load(file)

        analysis = KSLossSensitivityAnalysis()

        for res_obj in objs["results"]:
            del res_obj["integral_loss"]
            measured = []

            for meas in res_obj["measured"]:
                measured.append((meas["sparsity"], meas["loss"]))

            res_obj["measured"] = measured
            analysis.results.append(KSLossSensitivityResult(**res_obj))

        return analysis

    def __init__(self):
        self._results = []

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.dict())

    @property
    def results(self) -> List[KSLossSensitivityResult]:
        """
        :return: the individual results for the analysis
        """
        return self._results

    def get_result(self, name: str) -> KSLossSensitivityResult:
        """
        :param name: the layer name of the result to get
        :return: the result that matches the given name
        """
        for res in self._results:
            if res.name == name:
                return res

        raise ValueError("could not find name of {} in results".format(name))

    def results_summary(self, normalize: bool) -> Dict[str, Any]:
        layers, values = zip(*[(res.name, res.integral) for res in self._results])

        if normalize:
            mean = numpy.mean(values)
            std = numpy.std(values)
            values = [(val - mean) / std for val in values]

        return {
            "layers": layers,
            "values": values,
            "min": min(values),
            "max": max(values),
            "mean": numpy.mean(values),
            "std": numpy.std(values),
        }

    def dict(self) -> Dict[str, Any]:
        """
        :return: dictionary representation of the current instance
        """
        return {"results": [res.dict() for res in self._results]}

    def save_json(self, path: str):
        """
        :param path: the path to save the json file at representing the layer sensitivities
        """
        if not path.endswith(".json"):
            path += ".json"

        path = clean_path(path)
        create_parent_dirs(path)

        with open(path, "w") as file:
            json.dump(self.dict(), file)

    def plot(
        self, path: Union[str, None], normalize: bool = True, title: str = None,
    ) -> Union[Tuple[plt.Figure, plt.Axes], Tuple[None, None]]:
        """
        :param path: the path to save an img version of the chart, None to display the plot
        :param normalize: normalize the values to a unit distribution (0 mean, 1 std)
        :param title: the title to put on the chart
        :return: the created figure and axes if path is None, otherwise (None, None)
        """
        summary = self.results_summary(normalize)
        layers = summary["layers"]
        values = summary["values"]

        if normalize:
            mean = numpy.mean(values)
            std = numpy.std(values)
            values = [(val - mean) / std for val in values]

        height = round(len(layers) / 4) + 3
        fig = plt.figure(figsize=(12, height))
        ax = fig.add_subplot(111)

        if title is not None:
            ax.set_title(title)

        ax.invert_yaxis()
        frame = pandas.DataFrame(
            list(zip(layers, values)), columns=["Layer", "Sensitivity"]
        )
        frame.plot.barh(ax=ax, x="Layer", y="Sensitivity")
        plt.gca().invert_yaxis()

        if path is None:
            plt.show()

            return fig, ax

        path = clean_path(path)
        create_parent_dirs(path)
        plt.savefig(path)
        plt.close(fig)

        return None, None


class KSLossSensitivityProgress(object):
    """
    Simple class for tracking the progress of a sensitivity analysis
    """

    @staticmethod
    def standard_update_hook():
        """
        :return: a hook that will display a tqdm bar for tracking progress of the analysis
        """
        bar = None
        last_layer = None
        last_level = None

        def _update(progress: KSLossSensitivityProgress):
            nonlocal bar
            nonlocal last_layer
            nonlocal last_level

            if bar is None and last_layer is None and last_level is None:
                num_steps = len(progress.layers) * len(progress.sparsity_levels)
                print("num_steps: {}".format(num_steps))
                bar = auto.tqdm(total=num_steps, desc="KS Loss Sensitivity Analysis")
            elif bar is None:
                return

            if (
                (
                    last_layer is None
                    or last_layer != progress.layer_index
                    or last_level is None
                    or last_level != progress.sparsity_index
                )
                and progress.layer_index >= 0
                and progress.sparsity_index >= 0
            ):
                bar.update(1)
                last_layer = progress.layer_index
                last_level = progress.sparsity_index

            if progress.layer_index + 1 == len(
                progress.layers
            ) and progress.sparsity_index + 1 == len(progress.sparsity_levels):
                bar.close()
                bar = None

        return _update

    def __init__(
        self,
        layer_index: int,
        layer_name: str,
        layers: List[str],
        sparsity_index: int,
        sparsity_levels: List[float],
        measurement_step: int,
        samples_per_measurement: int,
    ):
        """
        :param layer_index: index of the current layer being evaluated, -1 for None
        :param layer_name: the name of the current layer being evaluated
        :param layers: a list of the layers that are being evaluated
        :param sparsity_index: index of the current sparsity level check for the current layer, -1 for None
        :param sparsity_levels: the sparsity levels to be checked for the current layer
        :param measurement_step: the current number of items processed for the measurement on the layer and sparsity lev
        :param samples_per_measurement: number of items to be processed for each layer and sparsity level
        """
        self.layer_index = layer_index
        self.layer_name = layer_name
        self.layers = layers
        self.sparsity_index = sparsity_index
        self.sparsity_levels = sparsity_levels
        self.measurement_step = measurement_step
        self.samples_per_measurement = samples_per_measurement

    def __repr__(self):
        return (
            "{}(layer_index={}, layer_name={}, layers={}, sparsity_index={}, sparsity_levels={},"
            " measurement_step={}, samples_per_measurement={})".format(
                self.__class__.__name__,
                self.layer_index,
                self.layer_name,
                self.layers,
                self.sparsity_index,
                self.sparsity_levels,
                self.measurement_step,
                self.samples_per_measurement,
            )
        )
