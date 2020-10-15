"""
Code related to the benchmark implementations for job workers
"""
from typing import Iterator, Dict, Any, Union, List
import itertools

import logging
from onnx import ModelProto

from neuralmagicML.onnx.utils import (
    DataLoader,
    NMModelRunner,
    ORTModelRunner,
    ModelRunner,
    check_load_model,
    get_ml_sys_info,
    get_node_by_id,
    prune_model_one_shot_iter,
)

from neuralmagicML.server.blueprints.utils import get_project_optimizer_by_ids

from neuralmagicML.server.models import (
    ProjectModel,
    BaseProjectProfile,
    ProjectBenchmark,
)
from neuralmagicML.server.schemas import (
    data_dump_and_validation,
    ProjectBenchmarkResultSchema,
    JobProgressSchema,
)
from neuralmagicML.server.workers.base import BaseJobWorker

_LOGGER = logging.getLogger(__name__)

__all__ = ["CreateBenchmarkJobWorker"]

NEURALMAGIC_ENGINE = "neural_magic"
ORT_CPU_ENGINE = "ort_cpu"
ORT_GPU_ENGINE = "ort_gpu"


class CreateBenchmarkJobWorker(BaseJobWorker):
    """
    A job worker for running and saving a benchmark for a given project
    and configuration.

    :param job_id: the id of the job this worker is running under
    :param project_id: the id of the project the worker is running for
    :param model_id: id of the model to run the loss profile for
    :param benchmark_id: the benchmark id that should be updated
    :param core_counts: list of core count to run on for benchmarking.
        -1 will use the maximum cores available
    :param batch_sizes: list of batch sizes to use for benchmarking
    :param instruction_sets: list of instruction sets
    :param inference_models: list of inference model to use for comparison with
        fields inference_engine and inference_model_optimization
    :param warmup_iterations_per_check: the number of warmup iterations to run for
        before checking performance / timing
    :param iterations_per_check: the number of iterations to run for each performance
        check / timing
    :return: the formatted args to be stored for later use
    """

    @classmethod
    def format_args(
        cls,
        model_id: str,
        benchmark_id: str,
        core_counts: List[int],
        batch_sizes: List[int],
        instruction_sets: List[str],
        inference_models: List[Dict[str, Union[str, None]]],
        warmup_iterations_per_check: int,
        iterations_per_check: int,
    ):
        """
        Format a given args into proper args to be stored for later use
        in the constructor for the job worker.

        :param model_id: id of the model to run the loss profile for
        :param benchmark_id: the benchmark id that should be updated
        :param core_counts: list of core count to run on for benchmarking.
            -1 will use the maximum cores available
        :param batch_sizes: list of batch sizes to use for benchmarking
        :param instruction_sets: list of instruction sets
        :param inference_models: list of inference model to use for comparison with
            fields inference_engine and inference_model_optimization
        :param warmup_iterations_per_check: the number of warmup iterations to run for before
            checking performance / timing
        :param iterations_per_check: the number of iterations to run for each performance
            check / timing
        :return: the formatted args to be stored for later use
        """
        return {
            "model_id": model_id,
            "benchmark_id": benchmark_id,
            "core_counts": core_counts,
            "batch_sizes": batch_sizes,
            "instruction_sets": instruction_sets,
            "inference_models": inference_models,
            "warmup_iterations_per_check": warmup_iterations_per_check,
            "iterations_per_check": iterations_per_check,
        }

    def __init__(
        self,
        job_id: str,
        project_id: str,
        model_id: str,
        benchmark_id: str,
        core_counts: List[int],
        batch_sizes: List[int],
        instruction_sets: List[str],
        inference_models: List[Dict[str, Union[str, None]]],
        warmup_iterations_per_check: int,
        iterations_per_check: int,
    ):
        super().__init__(job_id, project_id)
        self._model_id = model_id
        self._benchmark_id = benchmark_id
        self._core_counts = core_counts
        self._batch_sizes = batch_sizes
        self._instruction_sets = instruction_sets
        self._inference_models = inference_models
        self._warmup_iterations_per_check = warmup_iterations_per_check
        self._iterations_per_check = iterations_per_check

    @property
    def model_id(self) -> str:
        """
        :return: id of the model to run the loss profile for
        """
        return self._model_id

    @property
    def benchmark_id(self) -> str:
        """
        :return: id of the benchmark
        """
        return self._benchmark_id

    @property
    def core_counts(self) -> List[int]:
        """
        :return: list of core count to run on for benchmarking.
            -1 will use the maximum cores available
        """
        return self._core_counts

    @property
    def batch_sizes(self) -> List[int]:
        """
        :return: list of batch sizes to use for benchmarking
        """
        return self._batch_sizes

    @property
    def instruction_sets(self) -> List[str]:
        """
        :return: list of instruction sets
        """
        return self._instruction_sets

    @property
    def inference_models(self) -> List[Dict[str, str]]:
        """
        :return: list of inference model to use for comparison with
            fields inference_engine and inference_model_optimization
        """
        return self._inference_models

    @property
    def warmup_iterations_per_check(self) -> int:
        """
        :return: the number of warmup iterations to run for before checking
            performance / timing
        """
        return self._warmup_iterations_per_check

    @property
    def iterations_per_check(self) -> int:
        """
        :return: the number of iterations to run for each performance check / timing
        """
        return self._iterations_per_check

    def _get_project_model(self) -> ProjectModel:
        """
        :return: the project's model matching the given ids
        """
        model = ProjectModel.get_or_none(ProjectModel.model_id == self.model_id)

        if model is None:
            raise ValueError("could not find model_id of {}".format(self.model_id))

        return model

    def _get_project_benchmark(self) -> ProjectBenchmark:
        """
        :return: the project's benchmark matching the given ids
        """
        benchmark = ProjectBenchmark.get_or_none(
            ProjectBenchmark.benchmark_id == self.benchmark_id
        )

        if benchmark is None:
            raise ValueError(
                "could not find benchmark_id of {}".format(self.benchmark_id)
            )

        return benchmark

    def _get_pruned_model_proto(
        self, model_proto, inference_model_optimization: str
    ) -> ModelProto:
        _LOGGER.debug(
            "Pruning model with optim {}".format(inference_model_optimization)
        )

        optim = get_project_optimizer_by_ids(
            self.project_id, inference_model_optimization
        )
        nodes = []
        pruning_modifiers = optim.pruning_modifiers
        nodes = []
        sparsities = []

        for pruning_modifier in pruning_modifiers:
            for node in pruning_modifier.nodes:
                nodes.append(get_node_by_id(model_proto, node["node_id"]))
                sparsities.append(node["sparsity"] if node["sparsity"] else 0)
        for progress in prune_model_one_shot_iter(model_proto, nodes, sparsities):
            yield progress

    def _run_benchmark(
        self,
        benchmark: ProjectBenchmark,
        model: Union[str, ModelProto],
        runner: ModelRunner,
        core_count: int,
        batch_size: int,
        inference_engine: str,
        inference_model_optimization: Union[str, None],
        num_steps: int,
        step_index: int,
    ):
        data_iter = DataLoader.from_model_random(
            model, batch_size=batch_size, iter_steps=-1
        )

        measurements = []

        total_iterations = self.warmup_iterations_per_check + self.iterations_per_check

        iterations = 0
        for _, current_measurements in runner.run_iter(
            data_iter,
            show_progress=False,
            max_steps=total_iterations,
        ):
            measurements.append(current_measurements)
            iteration_percent = (iterations + 1) / (total_iterations)
            iter_val = (step_index + iteration_percent) / num_steps
            yield iter_val
            iterations += 1

        if self.warmup_iterations_per_check > 0:
            measurements = measurements[self.warmup_iterations_per_check :]

        result = data_dump_and_validation(
            ProjectBenchmarkResultSchema(),
            {
                "core_count": core_count,
                "batch_size": batch_size,
                "inference_engine": inference_engine,
                "inference_model_optimization": inference_model_optimization,
                "measurements": measurements,
            },
        )
        benchmark.result["benchmarks"].append(result)

    def run(self) -> Iterator[Dict[str, Any]]:
        """
        Perform the work for the job.
        Runs and saves the appropriate benchmark based on the configuration

        :return: an iterator containing progress update information
        """
        _LOGGER.info(
            (
                "running benchmark for project_id {} and "
                "model_id {} and benchmark_id {} with "
                "core_counts:{}, batch sizes:{} "
                "instruction_sets:{}, inference_models:{} "
            ).format(
                self.project_id,
                self.model_id,
                self.benchmark_id,
                self.core_counts,
                self.batch_sizes,
                self.instruction_sets,
                self.inference_models,
            )
        )

        project_model = self._get_project_model()
        project_model.validate_filesystem()
        benchmark = self._get_project_benchmark()
        benchmark.result = {"benchmarks": []}
        sys_info = get_ml_sys_info()
        cores_per_socket = (
            sys_info["cores_per_socket"] if "cores_per_socket" in sys_info else 1
        )
        num_sockets = sys_info["num_sockets"] if "num_sockets" in sys_info else 1
        max_cores = cores_per_socket * num_sockets

        optims = set()
        for inference_model in self.inference_models:
            inference_model_optimization = inference_model[
                "inference_model_optimization"
            ]
            if inference_model_optimization:
                optims.add(inference_model_optimization)

        iterables = [
            x
            for x in itertools.product(
                self.core_counts, self.batch_sizes, self.inference_models
            )
        ]
        num_steps = len(iterables) + len(optims)
        step_index = 0

        pruned_models = {}

        for inference_model_optimization in optims:
            model_proto = check_load_model(project_model.file_path)

            for progress in self._get_pruned_model_proto(
                model_proto, inference_model_optimization
            ):
                yield JobProgressSchema().dump(
                    {
                        "iter_indefinite": False,
                        "iter_class": "benchmark",
                        "iter_val": (step_index + progress) / num_steps,
                        "num_steps": num_steps,
                        "step_index": step_index,
                        "step_class": "pruning_{}".format(inference_model_optimization),
                    }
                )
            pruned_models[inference_model_optimization] = model_proto

            step_index += 1

        for core_count, batch_size, inference_model in iterables:
            inference_engine = inference_model["inference_engine"]
            inference_model_optimization = inference_model[
                "inference_model_optimization"
            ]

            model = project_model.file_path

            if inference_model_optimization:
                model = pruned_models[inference_model_optimization]

            if inference_engine == ORT_CPU_ENGINE and (
                core_count == max_cores or core_count < 1
            ):
                runner = ORTModelRunner(model, batch_size=batch_size)
            elif inference_engine == ORT_CPU_ENGINE and (
                core_count != max_cores and core_count > 0
            ):
                _LOGGER.error(
                    "Can only run onnxruntime with max core count of {}".format(
                        max_cores
                    )
                )
                raise Exception(
                    "Can only run onnxruntime with max core count of {}".format(
                        max_cores
                    )
                )
            elif inference_engine == NEURALMAGIC_ENGINE:
                runner = NMModelRunner(model, batch_size, core_count)
            elif inference_engine == ORT_GPU_ENGINE:
                raise NotImplementedError()
            else:
                raise ValueError("Invalid inference engine {}".format(inference_engine))

            step_class = (
                "{}_optim_{}_batch_size_{}_core_count_{}".format(
                    inference_engine,
                    inference_model_optimization,
                    batch_size,
                    core_count,
                )
                if inference_model_optimization
                else "{}_batch_size_{}_core_count_{}".format(
                    inference_engine, batch_size, core_count
                )
            )

            _LOGGER.debug(step_class)

            for progress in self._run_benchmark(
                benchmark,
                model,
                runner,
                core_count,
                batch_size,
                inference_engine,
                inference_model_optimization,
                num_steps,
                step_index,
            ):
                yield JobProgressSchema().dump(
                    {
                        "iter_indefinite": False,
                        "iter_class": "benchmark",
                        "iter_val": progress,
                        "num_steps": num_steps,
                        "step_index": step_index,
                        "step_class": step_class,
                    }
                )
            step_index += 1
        benchmark.save()
