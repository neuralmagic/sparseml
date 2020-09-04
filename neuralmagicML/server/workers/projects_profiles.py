"""
Job workers for running profiles within a project
"""

from typing import Iterator, Dict, Any, Union, List
import logging
import json

from neuralmagicML.onnx.utils import DataLoader
from neuralmagicML.onnx.recal import (
    approx_ks_loss_sensitivity_iter,
    one_shot_ks_loss_sensitivity_iter,
    one_shot_ks_perf_sensitivity_iter,
    KSSensitivityResult,
)
from neuralmagicML.server.schemas import (
    JobProgressSchema,
    ProjectProfileAnalysisSchema,
    ProjectProfileModelOpsMeasurementsSchema,
    ProjectProfileModelOpsBaselineMeasurementsSchema,
)
from neuralmagicML.server.models import (
    ProjectModel,
    BaseProjectProfile,
    ProjectLossProfile,
    ProjectPerfProfile,
)
from neuralmagicML.server.workers.base import BaseJobWorker


__all__ = [
    "CreateLossProfileJobWorker",
    "CreatePerfProfileJobWorker",
]


_LOGGER = logging.getLogger(__name__)


class BaseProfileJobWorker(BaseJobWorker):
    """
    Base job worker for working with profiles for projects

    :param job_id: the id of the job this worker is running under
    :param project_id: the id of the project the worker is running for
    :param model_id: id of the model to run the profile for
    :param profile_id: the profile id of the profile that should be updated
    """

    @classmethod
    def format_args(cls, **kwargs) -> Dict[str, Any]:
        """
        Format a given args into proper args to be stored for later use
        in the constructor for the job worker.

        :param kwargs: the args to format
        :return: the formatted args to be stored for later use
        """
        raise NotImplementedError()

    def __init__(self, job_id: str, project_id: str, model_id: str, profile_id: str):
        super().__init__(job_id, project_id)
        self._model_id = model_id
        self._profile_id = profile_id

    @property
    def model_id(self) -> str:
        """
        :return: id of the model to run the profile for
        """
        return self._model_id

    @property
    def profile_id(self) -> str:
        """
        :return: the profile id of the profile that should be updated
        """
        return self._profile_id

    def run(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError()

    def get_project_model(self) -> ProjectModel:
        """
        :return: the project's model matching the given ids
        """
        model = ProjectModel.get_or_none(ProjectModel.model_id == self.model_id)

        if model is None:
            raise ValueError("could not find model_id of {}".format(self.model_id))

        return model

    @staticmethod
    def add_profile_baseline_results(
        profile: BaseProjectProfile,
        results_model: KSSensitivityResult,
        results: List[KSSensitivityResult],
    ):
        """
        Helper to add baseline results to a profile's analysis

        :param profile: profile to add baseline results to
        :param results_model: the baseline results for the model
        :param results: the baseline results for the ops
        """
        model = {"measurement": results_model.baseline_average}
        ops = []

        for res in results:
            ops.append(
                {
                    "id": res.id_,
                    "name": res.name,
                    "index": res.index,
                    "measurement": res.baseline_average,
                }
            )

        profile.analysis[
            "baseline"
        ] = ProjectProfileModelOpsBaselineMeasurementsSchema().dump(
            {"model": model, "ops": ops}
        )

    @staticmethod
    def add_profile_pruning_results(
        profile: BaseProjectProfile,
        results_model: KSSensitivityResult,
        results: List[KSSensitivityResult],
    ):
        """
        Helper to add pruning results to a profile's analysis

        :param profile: profile to add pruning results to
        :param results_model: the pruning results for the model
        :param results: the pruning results for the ops
        """
        model = {
            "baseline_measurement_key": (str(results_model.baseline_measurement_key)),
            "measurements": {
                str(key): val for key, val in results_model.averages.items()
            },
        }
        ops = []

        for res in results:
            ops.append(
                {
                    "id": res.id_,
                    "name": res.name,
                    "index": res.index,
                    "baseline_measurement_key": (str(res.baseline_measurement_key)),
                    "measurements": {
                        str(key): val for key, val in res.averages.items()
                    },
                }
            )

        profile.analysis["pruning"] = ProjectProfileModelOpsMeasurementsSchema().dump(
            {"model": model, "ops": ops}
        )


class CreateLossProfileJobWorker(BaseProfileJobWorker):
    """
    A job worker for running and saving a loss profile for a given project
    and configuration.

    :param job_id: the id of the job this worker is running under
    :param project_id: the id of the project the worker is running for
    :param model_id: id of the model to run the profile for
    :param profile_id: the profile id of the profile that should be updated
    :param pruning_estimations: True to include pruning profile information
    :param pruning_estimation_type: loss analysis type to run,
        weight_magnitude or one_shot
    :param pruning_structure: type of pruning to use, (unstructured, block_4...)
    :param quantized_estimations: True to include quantized information in the profile
    """

    @classmethod
    def format_args(
        cls,
        model_id: str,
        profile_id: str,
        pruning_estimations: bool,
        pruning_estimation_type: str,
        pruning_structure: str,
        quantized_estimations: bool,
        **kwargs
    ) -> Union[None, Dict[str, Any]]:
        """
        Format a given args into proper args to be stored for later use
        in the constructor for the job worker.

        :param model_id: id of the model to run the loss profile for
        :param profile_id: the profile id of the loss profile that should be updated
        :param pruning_estimations: True to include pruning profile information
        :param pruning_estimation_type: loss analysis type to run,
            weight_magnitude or one_shot
        :param pruning_structure: type of pruning to use, (unstructured, block_4...)
        :param quantized_estimations: True to include quantized information
            in the profile, False otherwise
        :return: the formatted args to be stored for later use
        """
        return {
            "model_id": model_id,
            "profile_id": profile_id,
            "pruning_estimations": pruning_estimations,
            "pruning_estimation_type": pruning_estimation_type,
            "pruning_structure": pruning_structure,
            "quantized_estimations": quantized_estimations,
        }

    def __init__(
        self,
        job_id: str,
        project_id: str,
        model_id: str,
        profile_id: str,
        pruning_estimations: bool,
        pruning_estimation_type: str,
        pruning_structure: str,
        quantized_estimations: bool,
    ):
        super().__init__(job_id, project_id, model_id, profile_id)
        self._model_id = model_id
        self._profile_id = profile_id
        self._pruning_estimations = pruning_estimations
        self._pruning_estimation_type = pruning_estimation_type
        self._pruning_structure = pruning_structure
        self._quantized_estimations = quantized_estimations

    @property
    def model_id(self) -> str:
        """
        :return: id of the model to run the loss profile for
        """
        return self._model_id

    @property
    def profile_id(self) -> str:
        """
        :return: the profile id of the loss profile that should be updated
        """
        return self._profile_id

    @property
    def pruning_estimations(self) -> bool:
        """
        :return: True to include pruning profile information
        """
        return self._pruning_estimations

    @property
    def pruning_estimation_type(self) -> str:
        """
        :return: loss analysis type to run,
            weight_magnitude or one_shot
        """
        return self._pruning_estimation_type

    @property
    def pruning_structure(self) -> str:
        """
        :return: type of pruning to use, (unstructured, block_4...)
        """
        return self._pruning_structure

    @property
    def quantized_estimations(self) -> bool:
        """
        :return: True to include quantized information
            in the profile, False otherwise
        """
        return self._quantized_estimations

    def run(self) -> Iterator[Dict[str, Any]]:
        """
        Perform the work for the job.
        Runs and saves the appropriate loss profile based on the configuration

        :return: an iterator containing progress update information
        """
        _LOGGER.info(
            (
                "running loss profile for project_id {} and "
                "model_id {} and profile_id {} with "
                "pruning_estimations:{}, pruning_estimation_type:{}, "
                "pruning_structure:{}, quantized_estimations:{}"
            ).format(
                self.project_id,
                self.model_id,
                self.profile_id,
                self.pruning_estimations,
                self.pruning_estimation_type,
                self.pruning_structure,
                self.quantized_estimations,
            )
        )
        model = self.get_project_model()
        model.validate_filesystem()
        profile = self._get_project_loss_profile()
        profile.analysis = {}

        num_steps = 1
        if self.pruning_estimations:
            num_steps += 1
        if self.quantized_estimations:
            num_steps += 1

        for progress in self._run_baseline_loss(model, profile, num_steps):
            _LOGGER.debug(
                (
                    "loss profile baseline analysis for project_id {} and "
                    "model_id {} and profile_id {}: {}"
                ).format(self.project_id, self.model_id, self.profile_id, progress)
            )
            yield progress

        if self.pruning_estimations:
            if self.pruning_estimation_type == "weight_magnitude":
                for progress in self._run_weight_magnitude_pruning_sensitivity(
                    model, profile, num_steps
                ):
                    _LOGGER.debug(
                        (
                            "loss profile pruning weight magnitude analysis for "
                            "project_id {} and model_id {} and profile_id {}: {}"
                        ).format(
                            self.project_id, self.model_id, self.profile_id, progress
                        )
                    )
                    yield progress
            elif self.pruning_estimation_type == "one_shot":
                for progress in self._run_one_shot_pruning_sensitivity(
                    model, profile, num_steps
                ):
                    _LOGGER.debug(
                        (
                            "loss profile pruning one shot analysis for "
                            "project_id {} and model_id {} and profile_id {}: {}"
                        ).format(
                            self.project_id, self.model_id, self.profile_id, progress
                        )
                    )
                    yield progress
            else:
                raise ValueError(
                    "unrecognized pruning_estimation_type given of {}".format(
                        self.pruning_estimation_type
                    )
                )
        else:
            profile.analysis["pruning"] = None

        if self.quantized_estimations:
            raise NotImplementedError(
                "quantized estimations are currently not available"
            )
        else:
            profile.analysis["quantization"] = None

        profile.save()

    def _run_baseline_loss(
        self, model: ProjectModel, profile: ProjectLossProfile, num_steps: int
    ):
        analysis = None

        for (analysis, progress) in approx_ks_loss_sensitivity_iter(
            model.file_path, [0.0]
        ):
            yield JobProgressSchema().dump(
                {
                    "iter_indefinite": False,
                    "iter_class": "analysis",
                    "iter_val": progress.val,
                    "num_steps": num_steps,
                    "step_index": 0,  # baseline always runs first
                    "step_class": "baseline_estimation",
                }
            )

        # update but do not save until everything is completed so we don't hammer the DB
        CreateLossProfileJobWorker.add_profile_baseline_results(
            profile, analysis.results_model, analysis.results
        )

    def _run_weight_magnitude_pruning_sensitivity(
        self, model: ProjectModel, profile: ProjectLossProfile, num_steps: int
    ):
        if self.pruning_structure != "unstructured":
            raise ValueError(
                "pruning_structure of {} is not currently supported".format(
                    self.pruning_structure
                )
            )

        analysis = None

        for (analysis, progress) in approx_ks_loss_sensitivity_iter(model.file_path):
            yield JobProgressSchema().dump(
                {
                    "iter_indefinite": False,
                    "iter_class": "analysis",
                    "iter_val": progress.val,
                    "num_steps": num_steps,
                    "step_index": 1,  # baseline always runs before
                    "step_class": "pruning_estimation",
                }
            )

        # update but do not save until everything is completed so we don't hammer the DB
        CreateLossProfileJobWorker.add_profile_pruning_results(
            profile, analysis.results_model, analysis.results
        )

    def _run_one_shot_pruning_sensitivity(
        self, model: ProjectModel, profile: ProjectLossProfile, num_steps: int
    ):
        # TODO: fill in once data for a project can be used, random data won't correlate
        raise ValueError("one_shot loss sensitivity is not currently supported")

    def _get_project_loss_profile(self) -> ProjectLossProfile:
        loss_profile = ProjectLossProfile.get_or_none(
            ProjectLossProfile.profile_id == self._profile_id
        )

        if loss_profile is None:
            raise ValueError(
                "ProjectLossProfile with profile_id {} was not found".format(
                    self._profile_id
                )
            )

        return loss_profile


class CreatePerfProfileJobWorker(BaseProfileJobWorker):
    """
    A job worker for running and saving a perf profile for a given project
    and configuration.

    :param job_id: the id of the job this worker is running under
    :param project_id: the id of the project the worker is running for
    :param model_id: id of the model to run the profile for
    :param profile_id: the profile id of the profile that should be updated
    :param batch_size: batch size to use for perf analysis
    :param core_count: number of cores to run on for perf analysis. -1 will use
        the maximum cores available
    :param pruning_estimations: True to include pruning measurements
    :param quantized_estimations: True to include quantization measurements
    :param iterations_per_check: number of iterations of the batch size to
        run for each measurement check
    :param warmup_iterations_per_check: number of warmup iterations of the batch
        size to run before each measurement check
    """

    @classmethod
    def format_args(
        cls,
        model_id: str,
        profile_id: str,
        batch_size: int,
        core_count: int,
        pruning_estimations: bool,
        quantized_estimations: bool,
        iterations_per_check: int,
        warmup_iterations_per_check: int,
        **kwargs
    ) -> Union[None, Dict[str, Any]]:
        """
        Format a given args into proper args to be stored for later use
        in the constructor for the job worker.

        :param model_id: id of the model to run the loss profile for
        :param profile_id: the profile id of the loss profile that should be updated
        :param batch_size: batch size to use for perf analysis
        :param core_count: number of cores to run on for perf analysis.
            -1 will use the maximum cores available
        :param pruning_estimations: True to include pruning measurements
        :param quantized_estimations: True to include quantization measurements
        :param iterations_per_check: number of iterations of the batch size to
            run for each measurement check
        :param warmup_iterations_per_check: number of warmup iterations of the batch
            size to run before each measurement check
        :return: the formatted args to be stored for later use
        """
        return {
            "model_id": model_id,
            "profile_id": profile_id,
            "batch_size": batch_size,
            "core_count": core_count,
            "pruning_estimations": pruning_estimations,
            "quantized_estimations": quantized_estimations,
            "iterations_per_check": iterations_per_check,
            "warmup_iterations_per_check": warmup_iterations_per_check,
        }

    def __init__(
        self,
        job_id: str,
        project_id: str,
        model_id: str,
        profile_id: str,
        batch_size: int,
        core_count: int,
        pruning_estimations: bool,
        quantized_estimations: bool,
        iterations_per_check: int,
        warmup_iterations_per_check: int,
    ):
        super().__init__(job_id, project_id, model_id, profile_id)
        self._batch_size = batch_size
        self._core_count = core_count
        self._pruning_estimations = pruning_estimations
        self._quantized_estimations = quantized_estimations
        self._iterations_per_check = iterations_per_check
        self._warmup_iterations_per_check = warmup_iterations_per_check

    @property
    def batch_size(self) -> int:
        """
        :return: batch size to use for perf analysis
        """
        return self._batch_size

    @property
    def core_count(self) -> int:
        """
        :return: number of cores to run on for perf analysis.
            -1 will use the maximum cores available
        """
        return self._core_count

    @property
    def pruning_estimations(self) -> bool:
        """
        :return: True to include pruning profile information
        """
        return self._pruning_estimations

    @property
    def quantized_estimations(self) -> bool:
        """
        :return: True to include quantized information
            in the profile, False otherwise
        """
        return self._quantized_estimations

    @property
    def iterations_per_check(self) -> int:
        """
        :return: number of iterations of the batch size to
            run for each measurement check
        """
        return self._iterations_per_check

    @property
    def warmup_iterations_per_check(self):
        """
        :return: number of warmup iterations of the batch
            size to run before each measurement check
        """
        return self._warmup_iterations_per_check

    def run(self) -> Iterator[Dict[str, Any]]:
        """
        Perform the work for the job.
        Runs and saves the appropriate perf profile based on the configuration

        :return: an iterator containing progress update information
        """
        _LOGGER.info(
            (
                "running perf profile for project_id {} and "
                "model_id {} and profile_id {} with "
                "batch_size:{}, core_count:{}, "
                "pruning_estimations:{}, quantized_estimations:{}, "
                "iterations_per_check:{}, warmup_iterations_per_check:{}"
            ).format(
                self.project_id,
                self.model_id,
                self.profile_id,
                self.batch_size,
                self.core_count,
                self.pruning_estimations,
                self.quantized_estimations,
                self.iterations_per_check,
                self.warmup_iterations_per_check,
            )
        )
        model = self.get_project_model()
        model.validate_filesystem()
        profile = self._get_project_perf_profile()
        profile.analysis = {}
        data_loader = DataLoader.from_model_random(model.file_path, self.batch_size)

        num_steps = 1
        if self.pruning_estimations:
            num_steps += 1
        if self.quantized_estimations:
            num_steps += 1

        for progress in self._run_baseline_perf(model, data_loader, profile, num_steps):
            _LOGGER.debug(
                (
                    "perf profile baseline analysis for project_id {} and "
                    "model_id {} and profile_id {}: {}"
                ).format(self.project_id, self.model_id, self.profile_id, progress)
            )
            yield progress

        if self.pruning_estimations:
            for progress in self._run_pruning_sensitivity(
                model, data_loader, profile, num_steps
            ):
                _LOGGER.debug(
                    (
                        "perf profile pruning weight magnitude analysis for "
                        "project_id {} and model_id {} and profile_id {}: {}"
                    ).format(self.project_id, self.model_id, self.profile_id, progress)
                )
                yield progress
        else:
            profile.analysis["pruning"] = None

        if self.quantized_estimations:
            raise NotImplementedError(
                "quantized estimations are currently not available"
            )
        else:
            profile.analysis["quantization"] = None

        profile.save()

    def _run_baseline_perf(
        self,
        model: ProjectModel,
        data: DataLoader,
        profile: ProjectLossProfile,
        num_steps: int,
    ):
        analysis = None

        for (analysis, progress) in one_shot_ks_perf_sensitivity_iter(
            model.file_path,
            data,
            self.batch_size,
            self.core_count,
            self.iterations_per_check,
            self.warmup_iterations_per_check,
            sparsity_levels=[0.0],
            optimization_level=1,
        ):
            yield JobProgressSchema().dump(
                {
                    "iter_indefinite": False,
                    "iter_class": "analysis",
                    "iter_val": progress.val,
                    "num_steps": num_steps,
                    "step_index": 0,  # baseline always runs first
                    "step_class": "baseline_estimation",
                }
            )

        # update but do not save until everything is completed so we don't hammer the DB
        CreatePerfProfileJobWorker.add_profile_baseline_results(
            profile, analysis.results_model, analysis.results
        )

    def _run_pruning_sensitivity(
        self,
        model: ProjectModel,
        data: DataLoader,
        profile: ProjectLossProfile,
        num_steps: int,
    ):
        analysis = None

        for (analysis, progress) in one_shot_ks_perf_sensitivity_iter(
            model.file_path,
            data,
            self.batch_size,
            self.core_count,
            self.iterations_per_check,
            self.warmup_iterations_per_check,
            optimization_level=0,  # TODO: fix timings so optimization_level=1 works
            iters_sleep_time=0.1,  # hack for current issue with neuralmagic and GIL
        ):
            yield JobProgressSchema().dump(
                {
                    "iter_indefinite": False,
                    "iter_class": "analysis",
                    "iter_val": progress.val,
                    "num_steps": num_steps,
                    "step_index": 1,  # baseline always runs before
                    "step_class": "pruning_estimation",
                }
            )

        # update but do not save until everything is completed so we don't hammer the DB
        CreatePerfProfileJobWorker.add_profile_pruning_results(
            profile, analysis.results_model, analysis.results
        )

    def _get_project_perf_profile(self) -> ProjectPerfProfile:
        perf_profile = ProjectPerfProfile.get_or_none(
            ProjectPerfProfile.profile_id == self._profile_id
        )

        if perf_profile is None:
            raise ValueError(
                "ProjectPerfProfile with profile_id {} was not found".format(
                    self._profile_id
                )
            )

        return perf_profile
