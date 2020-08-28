"""
Job workers for running profiles within a project
"""

from typing import Iterator, Dict, Any, Union, List
import logging

from neuralmagicML.recal import (
    default_check_sparsities_loss,
    default_check_sparsities_perf,
    KSLossSensitivityAnalysis,
    KSPerfSensitivityAnalysis,
)
from neuralmagicML.onnx.utils import DataLoader
from neuralmagicML.onnx.recal import (
    iter_approx_ks_loss_sensitivity,
    iter_one_shot_ks_loss_sensitivity,
    iter_one_shot_ks_perf_sensitivity,
)
from neuralmagicML.server.schemas import (
    JobProgressSchema,
    CreateProjectLossProfileSchema,
    CreateProjectPerfProfileSchema,
    ProjectProfileAnalysisSchema,
    ProjectProfileOpBaselineSensitivitySchema,
    ProjectProfileOpSensitivitySchema,
)
from neuralmagicML.server.models import (
    database,
    ProjectLossProfile,
    ProjectPerfProfile,
)
from neuralmagicML.server.workers.base import BaseJobWorker

try:
    import neuralmagic
except:
    neuralmagic = None


__all__ = [
    "CreateLossProfileJobWorker",
    "CreatePerfProfileJobWorker",
]


_LOGGER = logging.getLogger(__name__)


class CreateLossProfileJobWorker(BaseJobWorker):
    """
    A job worker for running and saving a loss profile for a given project
    and configuration.

    :param job_id: the id of the job this worker is running under
    :param project_id: the id of the project the worker is running for
    :param profile_id: the profile id of the loss profile Model that should be updated
    :param model_path: file path to ONNX model to run loss profile for
    :param name: name of the profile
    :param pruning_estimations: True to include pruning profile information
    :param pruning_estimation_type: loss analysis type to run,
        weight_magnitude or one_shot
    :param pruning_structure: type of pruning to use, (unstructured, block_4...)
    :param quantized_estimations: True to include quantized information in the profile
    """

    @classmethod
    def format_args(
        cls,
        profile_id: str,
        model_path: str,
        name: str,
        pruning_estimations: bool,
        pruning_estimation_type: str,
        pruning_structure: str,
        quantized_estimations: bool,
        **kwargs
    ) -> Union[None, Dict[str, Any]]:
        """
        Format a given args into proper args to be stored for later use
        in the constructor for the job worker.

        :param profile_id: the profile id of the loss profile Model that should be updated
        :param model_path: file path to ONNX model to run loss profile for
        :param name: name of the profile
        :param pruning_estimations: True to include pruning profile information
        :param pruning_estimation_type: loss analysis type to run,
            weight_magnitude or one_shot
        :param pruning_structure: type of pruning to use, (unstructured, block_4...)
        :param quantized_estimations: True to incldue quantized information in the profile
        :return: the formatted args to be stored for later use
        """
        return {
            "profile_id": profile_id,
            "model_path": model_path,
            "name": name,
            "pruning_estimations": pruning_estimations,
            "pruning_estimation_type": pruning_estimation_type,
            "pruning_structure": pruning_structure,
            "quantized_estimations": quantized_estimations,
        }

    def __init__(
        self,
        job_id: str,
        project_id: str,
        profile_id: str,
        model_path: str,
        name: str,
        pruning_estimations: bool,
        pruning_estimation_type: str,
        pruning_structure: str,
        quantized_estimations: bool,
    ):
        super().__init__(job_id, project_id)
        self._profile_id = profile_id
        self._model_path = model_path
        self._name = name
        self._pruning_estimations = pruning_estimations
        self._pruning_estimation_type = pruning_estimation_type
        self._pruning_structure = pruning_structure
        self._quantized_estimations = quantized_estimations

    def run(self) -> Iterator[Dict[str, Any]]:
        """
        Perform the work for the job.
        Runs and saves the appropriate loss profile based on the configuration

        :return: an iterator containing progress update information
        """

        sparsity_levels = (
            default_check_sparsities_loss(True) if self._pruning_estimations else [0.0]
        )

        # run the appropriate analysis
        if self._pruning_estimation_type == "weight_magnitude":
            for progress in self._run_approx_loss_sensitivity(sparsity_levels):
                yield progress
        else:
            for progress in self._run_one_shot_loss_sensitivity(sparsity_levels):
                yield progress

    def _run_approx_loss_sensitivity(self, sparsity_levels: List[float]):
        # yield start progress to mark the expected flow
        yield JobProgressSchema().dump(
            {"iter_indefinite": False, "iter_class": "analysis", "iter_val": 0.0}
        )

        analysis_iter = iter_approx_ks_loss_sensitivity(
            self._model_path, sparsity_levels
        )
        loss_profile = self._get_project_loss_profile()
        for analysis, progress in analysis_iter:
            analysis = _parse_ks_analysis_profile(analysis, self._pruning_estimations)
            CreateLossProfileJobWorker._update_project_loss_profile(
                loss_profile, analysis
            )

            _LOGGER.info("updated profile {}".format(self._profile_id))
            yield JobProgressSchema().dump(
                {
                    "iter_indefinite": False,
                    "iter_class": "analysis",
                    "iter_val": progress,
                }
            )

    def _run_one_shot_loss_sensitivity(self, sparsity_levels: List[float]):
        # yield start progress to mark the expected flow
        yield JobProgressSchema().dump(
            {"iter_indefinite": False, "iter_class": "analysis", "iter_val": 0.0}
        )

        data_loader = DataLoader.from_model_random(self._model_path, 1)
        analysis_iter = iter_one_shot_ks_loss_sensitivity(
            self._model_path,
            data_loader,
            1,
            10,
            sparsity_levels=sparsity_levels,
            show_progress=False,
        )
        loss_profile = self._get_project_loss_profile()
        for analysis, progress in analysis_iter:
            analysis = _parse_ks_analysis_profile(analysis, self._pruning_estimations)
            CreateLossProfileJobWorker._update_project_loss_profile(
                loss_profile, analysis
            )

            _LOGGER.info("updated profile {}".format(self._profile_id))
            yield JobProgressSchema().dump(
                {
                    "iter_indefinite": False,
                    "iter_class": "analysis",
                    "iter_val": progress,
                }
            )

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

    @staticmethod
    def _update_project_loss_profile(
        loss_profile: ProjectLossProfile, analysis: ProjectProfileAnalysisSchema
    ):
        with database.atomic() as transaction:
            try:
                loss_profile.analysis = ProjectProfileAnalysisSchema().dump(analysis)
                loss_profile.save()
            except Exception as err:
                _LOGGER.error(
                    "error while uploading loss analysis results, rolling back: {}".format(
                        err
                    )
                )
                transaction.rollback()
                raise err


class CreatePerfProfileJobWorker(BaseJobWorker):
    """
    A job worker for running and saving a perf profile for a given project
    and configuration.

    :param job_id: the id of the job this worker is running under
    :param project_id: the id of the project the worker is running for
    :param profile_id: the profile id of the perf profile Model that should be updated
    :param model_path: file path to ONNX model to run perf profile for
    :param name: name of the profile to create
    :param batch_size: batch size to use for perf analysis
    :param core_count: number of cores to run on for perf analysis. -1 will use
        the maximum cores available
    :param pruning_estimations: True to include pruning measurements
    :param quantized_estimations: True to include quantization measurements
    """

    @classmethod
    def format_args(
        cls,
        profile_id: str,
        model_path: str,
        name: str,
        batch_size: int,
        core_count: int,
        pruning_estimations: bool,
        quantized_estimations: bool,
        **kwargs
    ) -> Union[None, Dict[str, Any]]:
        """
        Format a given args into proper args to be stored for later use
        in the constructor for the job worker.

        :param profile_id: the profile id of the perf profile Model that should be updated
        :param model_path: file path to ONNX model to run perf profile for
        :param name: name of the profile to create
        :param batch_size: batch size to use for perf analysis
        :param core_count: number of cores to run on for perf analysis. -1 will use
            the maximum cores available
        :param pruning_estimations: True to include pruning measurements
        :param quantized_estimations: True to include quantization measurements
        :return: the formatted args to be stored for later use
        """
        return {
            "profile_id": profile_id,
            "model_path": model_path,
            "name": name,
            "batch_size": batch_size,
            "core_count": core_count,
            "pruning_estimations": pruning_estimations,
            "quantized_estimations": quantized_estimations,
        }

    def __init__(
        self,
        job_id: str,
        project_id: str,
        profile_id: str,
        model_path: str,
        name: str,
        batch_size: int,
        core_count: int,
        pruning_estimations: bool,
        quantized_estimations: bool,
    ):
        super().__init__(job_id, project_id)
        self._profile_id = profile_id
        self._model_path = model_path
        self._name = name
        self._batch_size = batch_size
        self._core_count = core_count
        self._pruning_estimations = pruning_estimations
        self._quantized_estimations = quantized_estimations

    def run(self) -> Iterator[Dict[str, Any]]:
        """
        Perform the work for the job.
        Runs and saves the appropriate perf profile based on the configuration

        :return: an iterator containing progress update information
        """
        # yield start progress to mark the expected flow
        yield JobProgressSchema().dump(
            {"iter_indefinite": False, "iter_class": "analysis", "iter_val": 0.0}
        )

        data_loader = DataLoader.from_model_random(self._model_path, self._batch_size)
        sparsity_levels = (
            default_check_sparsities_perf() if self._pruning_estimations else [0.0]
        )
        analysis_iter = iter_one_shot_ks_perf_sensitivity(
            self._model_path,
            data_loader,
            self._batch_size,
            self._core_count,
            sparsity_levels=sparsity_levels,
            show_progress=False,
            wait_between_iters=True,
        )
        perf_profile = self._get_project_perf_profile()

        for analysis, progress in analysis_iter:
            analysis = _parse_ks_analysis_profile(analysis, self._pruning_estimations)
            CreatePerfProfileJobWorker._update_project_perf_profile(
                perf_profile, analysis
            )

            _LOGGER.info("updated profile {}".format(self._profile_id))
            yield JobProgressSchema().dump(
                {
                    "iter_indefinite": False,
                    "iter_class": "analysis",
                    "iter_val": progress,
                }
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

    @staticmethod
    def _update_project_perf_profile(
        perf_profile: ProjectPerfProfile, analysis: ProjectProfileAnalysisSchema
    ):
        # try to update instruction sets
        instruction_sets = None
        if neuralmagic is not None:
            instruction_sets = [
                ins.upper() for ins in neuralmagic.cpu.VALID_VECTOR_EXTENSIONS
            ]

        with database.atomic() as transaction:
            try:
                perf_profile.analysis = ProjectProfileAnalysisSchema().dump(analysis)
                if instruction_sets is not None:
                    perf_profile.instruction_sets = instruction_sets
                perf_profile.save()
            except Exception as err:
                _LOGGER.error(
                    "error while uploading perf analysis results, rolling back: {}".format(
                        err
                    )
                )
                transaction.rollback()
                raise err


def _parse_ks_analysis_profile(
    analysis: Union[KSLossSensitivityAnalysis, KSPerfSensitivityAnalysis],
    pruning_estimations: bool = True,
) -> ProjectProfileAnalysisSchema:
    baseline_results = []
    pruning_results = []

    for layer_results in analysis.dict()["results"]:
        baseline_result = {
            "id": layer_results["id"],
            "name": layer_results["name"],
            "index": layer_results["index"],
            "measurement": layer_results["baseline_average"],
        }
        baseline_results.append(
            ProjectProfileOpBaselineSensitivitySchema().dump(baseline_result)
        )

        if pruning_estimations:
            pruning_result = {
                "id": layer_results["id"],
                "name": layer_results["name"],
                "index": layer_results["index"],
                "baseline_measurement_index": layer_results[
                    "baseline_measurement_index"
                ],
                "measurements": {
                    str(k): v for k, v in layer_results["sparse_measurements"]
                },
            }
            pruning_results.append(
                ProjectProfileOpSensitivitySchema().dump(pruning_result)
            )

    profile_analysis = {
        "baseline": baseline_results,
        "pruning": pruning_results,
        "quantization": [],
    }
    return ProjectProfileAnalysisSchema().from_dict(profile_analysis)
