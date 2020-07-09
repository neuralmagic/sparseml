import argparse
import logging
import os
import json
import sys
from typing import Any, Callable, Dict, List

from flask import Flask
from flask_cors import CORS

from typing import List, Dict, Callable
from neuralmagicML.onnx import RecalProject, get_project_root, RecalConfig

__all__ = [
    "run_sparse_analysis_loss",
    "add_sparse_analysis_loss",
    "run_sparse_analysis_perf",
    "add_sparse_analysis_perf",
    "create_project",
    "create_config",
]

app = Flask(__name__, static_url_path="", template_folder="static")
CORS(app)

SPARSE_ANALYSIS_LOSS_COMMAND = "sparse-loss"
SPARSE_ANALYSIS_PERF_COMMAND = "sparse-perf"
CREATE_PROJECT_COMMAND = "create-project"
CREATE_CONFIG_COMMAND = "create-config"
ADD_SPARSE_ANALYSIS_LOSS_COMMAND = "add-sparse-loss"
ADD_SPARSE_ANALYSIS_PERF_COMMAND = "add-sparse-perf"
NM_STUDIO_SERVER_COMMAND = "server"


def _get_project(project_id: str, project_root: str):
    project_root = get_project_root(project_root)
    project_path = os.path.join(project_root, project_id)
    return RecalProject(project_path)


def _add_sparse_analysis(
    project: RecalProject, file_name: str, source: str, content: str, command: Callable,
):
    if source is None and content is None:
        raise Exception("Must specificy either source or content")
    elif source is not None:
        with open(source) as source_file:
            content = json.load(source_file)
    else:
        content = json.loads(content)

    results = command(file_name, content=content)
    return results


def run_sparse_analysis_loss(
    project_id: str,
    project_root: str,
    loss_file: str,
    batch_size: int,
    sparsity_levels: List[float],
    samples_per_measurement: int,
):
    """
    Runs sparse analysis loss. Requires neuralmagicML to be installed.

    :param project_id: Id of project to run
    :param project_root: Root folder where projects are saved
    :param loss_file: Loss file name where sparse analysis will be saved
    :param batch_size: the batch size of the inputs to be used with the model, default is 1
    :param sparsity_levels: Sparsity levels for sparse analysis
    :param samples_per_measurement: Samples per measurement to be ran
    :return: Loss Analysis in format:
    [
        {
            "baseline": {
                "sparsity": 0.0,
                "loss": 0.0
            },
            "id": NODE ID,
            "sparse": [{
                "sparsity": 0.4,
                "loss": 1.2
            },
            ...
            ]
        },
        ...
    ]
    """
    project = _get_project(project_id, project_root)

    results = project.run_sparse_analysis_loss(
        loss_file,
        batch_size=batch_size,
        sparsity_levels=sparsity_levels,
        samples_per_measurement=samples_per_measurement,
    )
    logging.info(
        f"Successfully saved sparse analysis at {project.loss_file_path(loss_file)}"
    )
    return results


def add_sparse_analysis_loss(
    project_id: str, project_root: str, loss_file: str, source: str, content: str
):
    """
    Saves a sparse analysis to a project from either an existing file or string representation

    :param project_id: Id of project to run
    :param project_root: Root folder where projects are saved
    :param loss_file: Loss file name where sparse analysis will be saved
    :param source: Source file to load sparse analysis
    :param content: Content to save to sparse analysis. Will be ignored if `source` is provided
    :return: Loss Analysis in format:
    [
        {
            "baseline": {
                "sparsity": 0.0,
                "loss": 0.0
            },
            "id": NODE ID,
            "sparse": [{
                "sparsity": 0.4,
                "loss": 1.2
            },
            ...
            ]
        },
        ...
    ]
    """
    project = _get_project(project_id, project_root)
    results = _add_sparse_analysis(
        project, loss_file, source, content, project.write_sparse_analysis_loss,
    )

    logging.info(
        f"Successfully saved sparse analysis at {project.loss_file_path(loss_file)}"
    )
    return results


def run_sparse_analysis_perf(
    project_id: str,
    project_root: str,
    perf_file: str,
    batch_size: int,
    sparsity_levels: List[float],
    optimization_level: int,
    num_cores: int,
    num_warmup_iterations: int,
    num_iterations: int,
):
    """
    Runs sparse analysis performance. Requires neuralmagic to be installed.

    :param project_id: Id of project to run
    :param project_root: Root folder where projects are saved
    :param perf_file: Perf file name where sparse analysis will be saved
    :param batch_size: the batch size of the inputs to be used with the model, default is 1
    :param sparsity_levels: Sparsity levels for sparse analysis
    :param optimization_level: how much optimization to perform, default is 1
    :param num_cores: the number of physical cores to run the model on, default is -1 (detect physical cores num)
    :param num_iterations: number of times to repeat execution, default is 1
    :param num_warmup_iterations: number of times to repeat unrecorded before starting actual benchmarking iterations
    :return: Perf Analysis in format:
    [
        {
            "baseline": {
                "sparsity": 0.0,
                "flops": 1000,
                "timing": 1.0
            },
            "id": NODE ID,
            "sparse": [{
                "sparsity": 0.4,
                "flops": 1000,
                "timing": 0.9
            },
            ...
            ]
        },
        ...
    ]
    """
    project = _get_project(project_id, project_root)

    results = project.run_sparse_analysis_perf(
        perf_file,
        batch_size=batch_size,
        sparsity_levels=sparsity_levels,
        optimization_level=optimization_level,
        num_cores=num_cores,
        num_warmup_iterations=num_warmup_iterations,
        num_iterations=num_iterations,
    )

    logging.info(
        f"Successfully saved sparse analysis at {project.perf_file_path(perf_file)}"
    )
    return results


def add_sparse_analysis_perf(
    project_id: str, project_root: str, perf_file: str, source: str, content: str
):
    """
    Saves a sparse analysis to a project from either an existing file or string representation

    :param project_id: Id of project to run
    :param project_root: Root folder where projects are saved
    :param perf_file: Perf file name where sparse analysis will be saved
    :param source: Source file to load sparse analysis
    :param content: Content to save to sparse analysis. Will be ignored if `source` is provided
    :return: Perf Analysis in format:
    [
        {
            "baseline": {
                "sparsity": 0.0,
                "flops": 1000,
                "timing": 1.0
            },
            "id": NODE ID,
            "sparse": [{
                "sparsity": 0.4,
                "flops": 1000,
                "timing": 0.9
            },
            ...
            ]
        },
        ...
    ]
    """
    project = _get_project(project_id, project_root)
    results = _add_sparse_analysis(
        project, loss_file, source, content, project.write_sparse_analysis_perf,
    )

    logging.info(
        f"Successfully saved sparse analysis at {project.perf_file_path(loss_file)}"
    )
    return results


def create_project(
    model_path: str, project_root: str, project_name: str,
):
    """
    Creates a new project using model

    :param model_path: Path of onnx model
    :param project_root: Root folder where projects are saved
    :param project_name: Name of project
    :return: Project Config:
    {
        "projectId": 12345-67890
        "projectName": "test"
    }
    """
    config = RecalProject.register_project(
        model_path, {"projectName": project_name}, project_root=project_root
    )
    logging.info(f"Created project with ID {config.id}")
    return config.config_settings


def create_config(
    project_id: str,
    project_root: str,
    config_file: str,
    pruning_profile: str,
    sparsities: List[float],
    pruning_update_frequency: float,
    loss_analysis: str,
    perf_analysis: str,
    training_epochs: int,
    stabilization_epochs: int,
    pruning_epochs: int,
    fine_tuning_epochs: int,
    framework: str,
    init_pruning_sparsity: float,
    init_lr: float,
    init_training_lr: float,
    final_training_lr: float,
    fine_tuning_gamma: float,
):
    project = _get_project(project_id, project_root)
    config = RecalConfig.create_config(
        project,
        pruning_profile=pruning_profile,
        sparsities=sparsities,
        pruning_update_frequency=pruning_update_frequency,
        loss_analysis=loss_analysis,
        perf_analysis=perf_analysis,
        training_epochs=training_epochs,
        stabilization_epochs=stabilization_epochs,
        pruning_epochs=pruning_epochs,
        fine_tuning_epochs=fine_tuning_epochs,
        framework=framework,
        init_pruning_sparsity=init_pruning_sparsity,
        init_lr=init_lr,
        config_file=config_file,
        init_training_lr=init_training_lr,
        final_training_lr=final_training_lr,
        fine_tuning_gamma=fine_tuning_gamma,
    )
    config.save()


def _add_project_root_parser(parser):
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Root directory where files are stored",
    )


def _add_project_parser(parser):
    parser.add_argument("project_id", type=str, help="Project ID")
    _add_project_root_parser(parser)


def _add_sparsity_level_parser(parser):
    parser.add_argument(
        "--sparsity-levels", "-s", type=float, nargs="+", help="List of sparsity levels"
    )


def _add_batch_size_parser(parser):
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size of model input. Default 1",
    )


def _subparser_sparse_loss(subparsers):
    """
    Running sparse perf command subparser

    e.g.
    sparse-perf 12345-67890 --loss-file default --samples-per-measurement 5 \
        --batch-size 4 --sparsity-levels 0 0.4 0.6 0.7 0.9

    :param subparsers: ArgumentParser handling subparsing
    """
    subparser = subparsers.add_parser(
        SPARSE_ANALYSIS_LOSS_COMMAND, help="Run sparse perf analysis"
    )

    _add_project_parser(subparser)
    _add_sparsity_level_parser(subparser)
    _add_batch_size_parser(subparser)

    subparser.add_argument(
        "--loss-file",
        "-f",
        type=str,
        required=True,
        help="File name to save loss analysis",
    )

    subparser.add_argument(
        "--samples-per-measurement",
        "-m",
        type=int,
        default=5,
        help="Amount of batches to run when running sparse analysis. Default 5",
    )


def _subparser_add_sparse_loss(subparsers):
    """
    Add sparse loss command subparser

    e.g.
    If reading from a file
    add-sparse-loss 12345-67890 --loss-file default -src ~/Downloads/loss.yaml

    If using content
    add-sparse-loss 12345-67890 --loss-file default --content "{ 'test': 'test' }"

    :param subparsers: ArgumentParser handling subparsing
    """
    subparser = subparsers.add_parser(
        ADD_SPARSE_ANALYSIS_LOSS_COMMAND, help="Add sparse perf analysis"
    )

    _add_project_parser(subparser)

    subparser.add_argument(
        "--loss-file",
        "-f",
        type=str,
        required=True,
        help="File name to save loss analysis",
    )

    subparser.add_argument(
        "--source",
        "-src",
        type=str,
        default=None,
        help="Source file to read analysis from",
    )
    subparser.add_argument(
        "--content",
        type=str,
        default=None,
        help="Content to save to analysis. Will be ignored if 'source' is provided",
    )


def _subparser_sparse_perf(subparsers):
    """
    Running sparse perf command subparser

    e.g.
    sparse-perf 12345-67890 -perf-file default --optimization-level 0 --num-cores 4 \
        --num-warmup-iterations 5 --num-iterations 30 --batch-size 4 \
        --sparsity-levels 0 0.4 0.6 0.7 0.9

    :param subparsers: ArgumentParser handling subparsing
    """
    subparser = subparsers.add_parser(
        SPARSE_ANALYSIS_PERF_COMMAND, help="Run sparse perf analysis"
    )

    _add_project_parser(subparser)
    _add_sparsity_level_parser(subparser)
    _add_batch_size_parser(subparser)

    subparser.add_argument(
        "--perf-file",
        "-f",
        type=str,
        required=True,
        help="File name to save perf analysis",
    )

    subparser.add_argument(
        "--optimization-level",
        "-o",
        type=int,
        default=0,
        help="Level of optimization in benchmarking",
    )

    subparser.add_argument(
        "--num-cores",
        "-c",
        type=int,
        default=None,
        help="Number of cores used for benchmarking",
    )

    subparser.add_argument(
        "--num-warmup-iterations",
        "-w",
        type=int,
        default=5,
        help="Number of warm up iterations before benchmarking starts",
    )

    subparser.add_argument(
        "--num-iterations",
        "-i",
        type=int,
        default=30,
        help="Number of iterations to run for benchmarking",
    )


def _subparser_add_sparse_perf(subparsers):
    """
    Add sparse perf command subparser

    e.g.
    If reading from a file
    add-sparse-perf 12345-67890 --perf-file default -src ~/Downloads/perf.yaml

    If using content
    add-sparse-perf 12345-67890 -perf-file default --content "{ 'test': 'test' }"

    :param subparsers: ArgumentParser handling subparsing
    """
    subparser = subparsers.add_parser(
        ADD_SPARSE_ANALYSIS_PERF_COMMAND, help="Add sparse perf analysis"
    )

    _add_project_parser(subparser)

    subparser.add_argument(
        "--perf-file",
        "-f",
        type=str,
        required=True,
        help="File name to save perf analysis",
    )

    subparser.add_argument(
        "--source",
        "-src",
        type=str,
        default=None,
        help="Source file to read analysis from",
    )
    subparser.add_argument(
        "--content",
        type=str,
        default=None,
        help="Content to save to analysis. Will be ignored if 'source' is provided",
    )


def _subparser_create_project(subparsers):
    """
    Create project command subparser

    e.g.
    create-project --model-path ~/Downloads/model.onnx --project-name test_project

    :param subparsers: ArgumentParser handling subparsing
    """
    subparser = subparsers.add_parser(CREATE_PROJECT_COMMAND, help="Create projects")

    subparser.add_argument(
        "--model-path", type=str, required=True, help="Path of source model"
    )
    subparser.add_argument(
        "--project-name", type=str, required=True, help="Name of project"
    )

    _add_project_root_parser(subparser)


def _subparser_create_config(subparsers):
    """
    Running create config subparser

    e.g.
    create-config 12345-67890 --config-file test.yaml --pruning-profile loss --loss-analysis one_shot
        --sparsities 0.8 0.85 0.9 --training-epochs 50 --init-training-lr 0.05 --final-training-lr 0.0001

    :param subparsers: ArgumentParser handling subparsing
    """
    subparser = subparsers.add_parser(CREATE_CONFIG_COMMAND, help="Create recal config for pruning")

    _add_project_parser(subparser)

    subparser.add_argument(
        "--config-file", type=str, required=True, help="Name of config file"
    )

    subparser.add_argument(
        "--loss-analysis",
        type=str,
        help="Name of loss analysis file created with sparse-loss command",
    )

    subparser.add_argument(
        "--perf-analysis",
        type=str,
        help="Name of perf analysis file created with sparse-perf command",
    )

    subparser.add_argument(
        "--pruning-profile",
        type=str,
        required=True,
        help="Pruning profile. One of uniform, loss, performance, or balanced",
    )

    subparser.add_argument(
        "--sparsities",
        type=float,
        nargs="+",
        required=True,
        help="List of sparsity levels for each bin from low pruning to high pruning",
    )

    subparser.add_argument(
        "--pruning-update-frequency",
        type=float,
        help="Frequency of pruning at each epoch",
        default=1,
    )

    subparser.add_argument(
        "--init-lr",
        type=float,
        help="Learning rate at the start of pruning. Remains constant until fine tuning",
    )

    subparser.add_argument(
        "--init-pruning-sparsity",
        type=float,
        default=0.05,
        help="Initial sparsity for pruning. Default 0.05",
    )

    subparser.add_argument(
        "--training-epochs",
        type=int,
        help="Number of training epochs the model was trained on. Must either provide "
        + "training_epochs or provide all of stabilization_epochs, pruning_epochs, fine_tuning_epochs",
    )

    subparser.add_argument(
        "--stabilization-epochs",
        type=int,
        help="Number of stabilization epochs for pruning. Must either provide "
        + "training_epochs or provide all of stabilization_epochs, pruning_epochs, fine_tuning_epochs",
    )

    subparser.add_argument(
        "--pruning-epochs",
        type=int,
        help="Number of pruning epochs for pruning. Must either provide "
        + "training_epochs or provide all of stabilization_epochs, pruning_epochs, fine_tuning_epochs",
    )

    subparser.add_argument(
        "--fine-tuning-epochs",
        type=int,
        help="Number of fine tuning epochs for pruning. Must either provide "
        + "training_epochs or provide all of stabilization_epochs, pruning_epochs, fine_tuning_epochs",
    )

    subparser.add_argument(
        "--framework",
        type=str,
        default="pytorch",
        help="Framework used for config. Defaults to pytorch",
    )

    subparser.add_argument(
        "--init-training-lr", type=float, help="Initial learning rate used in training"
    )
    subparser.add_argument(
        "--final-training-lr", type=float, help="Final learning rate used in training"
    )
    subparser.add_argument(
        "--fine-tuning-gamma",
        type=float,
        default=0.2,
        help="Factor to decrease learning rate at each step of fine tuning",
    )


def _subparser_flask(subparsers):
    """
    Run flask server

    e.g.
    server 12345-67890 --project-root ~/nm-projects --host 127.0.0.1 --port 7899 --logging-level WARNING
    :param subparsers: ArgumentParser handling subparsing
    """
    subparser = subparsers.add_parser(
        NM_STUDIO_SERVER_COMMAND, help="Perfboard Server"
    )

    subparser.add_argument("--project-root", default=None)
    subparser.add_argument("--host", default="0.0.0.0", type=str)
    subparser.add_argument("--port", default=7890, type=int)
    subparser.add_argument("--logging-level", default="WARNING", type=str)


def main():
    parser = argparse.ArgumentParser(prog="Perfboard")

    subparsers = parser.add_subparsers(
        help="available commands for perfboard", dest="command"
    )

    _subparser_add_sparse_loss(subparsers)
    _subparser_add_sparse_perf(subparsers)
    _subparser_create_project(subparsers)
    _subparser_create_config(subparsers)
    _subparser_flask(subparsers)
    _subparser_sparse_loss(subparsers)
    _subparser_sparse_perf(subparsers)

    parse_args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    kwargs = vars(parse_args)
    command = parse_args.command
    del kwargs["command"]

    if command == NM_STUDIO_SERVER_COMMAND:
        raise Exception(f"Command {NM_STUDIO_SERVER_COMMAND} currently not implimented")

    elif command == SPARSE_ANALYSIS_LOSS_COMMAND:
        run_sparse_analysis_loss(**kwargs)

    elif command == SPARSE_ANALYSIS_PERF_COMMAND:
        run_sparse_analysis_perf(**kwargs)

    elif command == CREATE_PROJECT_COMMAND:
        create_project(**kwargs)

    elif command == CREATE_CONFIG_COMMAND:
        create_config(**kwargs)

    elif command == ADD_SPARSE_ANALYSIS_LOSS_COMMAND:
        add_sparse_analysis_loss(**kwargs)

    elif command == ADD_SPARSE_ANALYSIS_PERF_COMMAND:
        add_sparse_analysis_perf(**kwargs)

    else:
        raise Exception(f"No command provided. List of commands: ")


if __name__ == "__main__":
    main()
