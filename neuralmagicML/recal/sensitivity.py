"""
Generic code related to sensitivity analysis.
"""

from typing import Tuple, List, Dict, Any, Union
import json
import numpy
import matplotlib.pyplot as plt
import pandas

from neuralmagicML.utils.helpers import (
    clean_path,
    create_parent_dirs,
    interpolated_integral,
)


__all__ = [
    "default_check_sparsities",
    "KSLossSensitivityAnalysis",
    "LRLossSensitivityAnalysis",
]


def default_check_sparsities(extended: bool) -> Tuple[float, ...]:
    if not extended:
        return 0.0, 0.05, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99

    sparsities = [float(s) / 100.0 for s in range(100)]

    return tuple(sparsities)


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
            analysis.add_result(
                res_obj["param"], res_obj["index"], res_obj["sparse_measurements"]
            )

        return analysis

    def __init__(self):
        self._results = []

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.dict())

    @property
    def results(self) -> List[Dict[str, Any]]:
        """
        :return: the individual results for the analysis
        """
        return self._results

    def add_result(
        self,
        param: str,
        index: int,
        sparse_measurements: List[Tuple[float, List[float]]],
    ):
        """
        Add a result to the sensitivity analysis for a specific param

        :param param: the param to add the result for
        :param index: the index of the param as found in the model parameters
        :param sparse_measurements: a list of measurements each made up of
            a tuple of (sparsity, losses)
        """
        sparse_averages = [
            (ks, numpy.mean(meas).item()) for ks, meas in sparse_measurements
        ]
        sparse_loss_avg = numpy.mean([avg for ks, avg in sparse_averages])
        sparse_loss_integral = interpolated_integral(sparse_averages)

        self._results.append(
            {
                "param": param,
                "index": index,
                "sparse_measurements": sparse_measurements,
                "sparse_averages": sparse_averages,
                "sparse_loss_avg": sparse_loss_avg,
                "sparse_loss_integral": sparse_loss_integral,
            }
        )

    def get_result(self, param: str) -> Dict[str, Any]:
        """
        get a result from the sensitivity analysis for a specific param

        :param param: the param to get the result for
        :return: a dictionary containing the sensitivity results for the param
        """
        for res in self._results:
            if param == res["param"]:
                return res

        raise ValueError("could not find param {} in results".format(param))

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
            json.dump(self.dict(), file)

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
        params = [res["param"] for res in self._results]
        values = [
            res["sparse_loss_integral"] if plot_integral else res["sparse_loss_avg"]
            for res in self._results
        ]

        if normalize:
            mean = numpy.mean(values)
            std = numpy.std(values)
            values = [(val - mean) / std for val in values]

        height = round(len(params) / 4) + 3
        fig = plt.figure(figsize=(12, height))
        ax = fig.add_subplot(111)

        if title is not None:
            ax.set_title(title)

        ax.invert_yaxis()
        frame = pandas.DataFrame(
            list(zip(params, values)), columns=["Layer", "Sensitivity"]
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

    def print_res(self):
        """
        Print the recorded sensitivity values results
        """

        print("KS Sensitivity")
        print("\tLoss Avg\t\tLoss Int\t\tParam")

        for res in self._results:
            print(
                "\t{:.4f}\t\t{:.4f}\t\t{}".format(
                    res["sparse_loss_avg"], res["sparse_loss_integral"], res["param"]
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
            json.dump(self.dict(), file)

    def plot(
        self, path: Union[str, None], title: str = None,
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
