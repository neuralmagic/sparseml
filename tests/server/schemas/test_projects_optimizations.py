from typing import Dict, List, Union
import pytest
from datetime import datetime

from neuralmagicML.server.schemas.projects_optimizations import (
    ProjectAvailableModelModificationsSchema,
    ProjectOptimizationModifierPruningNodeSchema,
    ProjectOptimizationModifierPruningSchema,
    ProjectOptimizationModifierQuantizationNodeSchema,
    ProjectOptimizationModifierQuantizationSchema,
    ProjectOptimizationModifierLRSchema,
    ProjectOptimizationModifierLRSetArgsSchema,
    ProjectOptimizationModifierLRStepArgsSchema,
    ProjectOptimizationModifierLRMultiStepArgsSchema,
    ProjectOptimizationModifierLRExponentialArgsSchema,
    ProjectOptimizationModifierLRScheduleSchema,
    ProjectOptimizationModifierTrainableSchema,
    ProjectOptimizationModifierTrainableNodeSchema,
    ProjectOptimizationSchema,
    GetProjectOptimizationBestEstimatedResultsSchema,
    CreateProjectOptimizationSchema,
    UpdateProjectOptimizationSchema,
    CreateUpdateProjectOptimizationModifiersPruningSchema,
    CreateUpdateProjectOptimizationModifiersQuantizationSchema,
    CreateUpdateProjectOptimizationModifiersLRScheduleSchema,
    CreateUpdateProjectOptimizationModifiersTrainableSchema,
    SearchProjectOptimizationsSchema,
    ResponseProjectOptimizationFrameworksAvailableSchema,
    ResponseProjectOptimizationFrameworksAvailableSamplesSchema,
    ResponseProjectOptimizationModifiersAvailable,
    ResponseProjectOptimizationModifiersBestEstimated,
    ResponseProjectOptimizationSchema,
    ResponseProjectOptimizationsSchema,
    ResponseProjectOptimizationDeletedSchema,
    ResponseProjectOptimizationModifierDeletedSchema,
)
from tests.server.helper import schema_tester


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "pruning": False,
                "quantization": False,
                "sparse_transfer_learning": False,
            },
            {
                "pruning": False,
                "quantization": False,
                "sparse_transfer_learning": False,
            },
        )
    ],
)
def test_project_available_model_modifications_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectAvailableModelModificationsSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "node_id": None,
                "sparsity": None,
                "est_recovery": None,
                "est_perf_gain": None,
                "est_time": None,
                "est_time_baseline": None,
                "est_loss_sensitivity": None,
                "est_perf_sensitivity": None,
                "flops": None,
                "flops_baseline": None,
                "params": None,
                "params_baseline": None,
                "compression": None,
                "est_time_gain": None,
                "flops_gain": None,
                "overriden": None,
                "perf_sensitivities": [],
                "loss_sensitivities": [],
                "overridden": False,
            },
            {
                "perf_sensitivities": [],
                "overridden": False,
                "params_baseline": None,
                "flops_baseline": None,
                "est_recovery": None,
                "est_perf_sensitivity": None,
                "compression": None,
                "loss_sensitivities": [],
                "est_loss_sensitivity": None,
                "flops_gain": None,
                "params": None,
                "flops": None,
                "est_time_baseline": None,
                "node_id": None,
                "sparsity": None,
                "est_time": None,
                "est_time_gain": None,
            },
        ),
        (
            {
                "est_recovery": 1.0,
                "est_time": None,
                "params_baseline": 864.0,
                "est_time_baseline": None,
                "params": 864.0,
                "compression": 1.0,
                "flops_baseline": 10838016.0,
                "node_id": "165",
                "overridden": True,
                "perf_sensitivities": [
                    (0.0, 0.0),
                    (0.2, 2167603.1999999993),
                    (0.4, 4335206.4),
                    (0.6, 6502809.6),
                    (0.8, 8670412.8),
                    (0.9, 9754214.4),
                    (0.95, 10296115.2),
                    (0.99, 10729635.84),
                ],
                "est_loss_sensitivity": 0.4556600749492645,
                "flops": 10838016.0,
                "flops_gain": 1.0,
                "loss_sensitivities": [
                    (0.0, 0.0),
                    (0.2, 2.1940615828070636e-13),
                    (0.4, 3.5482580415524545e-12),
                    (0.6, 0.024138422682881355),
                    (0.8, 0.16036564111709595),
                    (0.9, 0.3132839798927307),
                    (0.95, 0.4556600749492645),
                    (0.99, 0.6454770565032959),
                ],
                "sparsity": None,
                "est_perf_sensitivity": 10296115.2,
                "est_time_gain": None,
            },
            {
                "est_recovery": 1.0,
                "est_time": None,
                "params_baseline": 864.0,
                "est_time_baseline": None,
                "params": 864.0,
                "compression": 1.0,
                "flops_baseline": 10838016.0,
                "node_id": "165",
                "overridden": True,
                "perf_sensitivities": [
                    (0.0, 0.0),
                    (0.2, 2167603.1999999993),
                    (0.4, 4335206.4),
                    (0.6, 6502809.6),
                    (0.8, 8670412.8),
                    (0.9, 9754214.4),
                    (0.95, 10296115.2),
                    (0.99, 10729635.84),
                ],
                "est_loss_sensitivity": 0.4556600749492645,
                "flops": 10838016.0,
                "flops_gain": 1.0,
                "loss_sensitivities": [
                    (0.0, 0.0),
                    (0.2, 2.1940615828070636e-13),
                    (0.4, 3.5482580415524545e-12),
                    (0.6, 0.024138422682881355),
                    (0.8, 0.16036564111709595),
                    (0.9, 0.3132839798927307),
                    (0.95, 0.4556600749492645),
                    (0.99, 0.6454770565032959),
                ],
                "sparsity": None,
                "est_perf_sensitivity": 10296115.2,
                "est_time_gain": None,
            },
        ),
    ],
)
def test_project_optimization_modifier_pruning_node_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectOptimizationModifierPruningNodeSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": None,
                "end_epoch": None,
                "update_frequency": None,
                "mask_type": "unstructured",
                "sparsity": None,
                "balance_perf_loss": 0,
                "filter_min_sparsity": None,
                "filter_min_perf_gain": None,
                "filter_min_recovery": None,
                "nodes": [],
                "est_loss_sensitivity": None,
                "est_perf_sensitivity": None,
                "flops_gain": None,
                "est_time_gain": None,
                "compression": None,
                "est_recovery": None,
                "est_perf_gain": None,
                "est_time": None,
                "est_time_baseline": None,
                "flops": None,
                "flops_baseline": None,
                "params": None,
                "params_baseline": None,
            },
            {
                "start_epoch": None,
                "modifier_id": "modifier id",
                "filter_min_sparsity": None,
                "optim_id": "optim id",
                "filter_min_recovery": None,
                "mask_type": "unstructured",
                "filter_min_perf_gain": None,
                "est_time_baseline": None,
                "flops_gain": None,
                "est_time": None,
                "update_frequency": None,
                "sparsity": None,
                "params": None,
                "est_loss_sensitivity": None,
                "est_perf_sensitivity": None,
                "balance_perf_loss": 0.0,
                "nodes": [],
                "flops_baseline": None,
                "compression": None,
                "params_baseline": None,
                "flops": None,
                "est_recovery": None,
                "est_time_gain": None,
                "end_epoch": None,
            },
            None,
        ),
        (
            {
                "balance_perf_loss": 1.0,
                "compression": 23.070087763135966,
                "created": "2020-10-29T19:02:39.977360",
                "end_epoch": 34.0,
                "est_loss_sensitivity": 0.20195244026503392,
                "est_perf_sensitivity": 19331324.442857143,
                "est_recovery": 0.6253260677145712,
                "est_time": None,
                "est_time_baseline": None,
                "est_time_gain": None,
                "filter_min_perf_gain": 0.75,
                "filter_min_recovery": -1.0,
                "filter_min_sparsity": 0.4,
                "flops": 97905673.52,
                "flops_baseline": 569765352.0,
                "flops_gain": 5.819533552196129,
                "mask_type": "unstructured",
                "modified": "2020-10-29T19:02:40.818771",
                "modifier_id": "efbbd23e81c04129a916c5da5572df0e",
                "nodes": [
                    {
                        "compression": 1.0,
                        "est_loss_sensitivity": 0.4556600749492645,
                        "est_perf_sensitivity": 10296115.2,
                        "est_recovery": 1.0,
                        "est_time": None,
                        "est_time_baseline": None,
                        "est_time_gain": None,
                        "flops": 10838016.0,
                        "flops_baseline": 10838016.0,
                        "flops_gain": 1.0,
                        "loss_sensitivities": [
                            [0.0, 0.0],
                            [0.2, 2.1940615828070636e-13],
                            [0.4, 3.5482580415524545e-12],
                            [0.6, 0.024138422682881355],
                            [0.8, 0.16036564111709595],
                            [0.9, 0.3132839798927307],
                            [0.95, 0.4556600749492645],
                            [0.99, 0.6454770565032959],
                        ],
                        "node_id": "165",
                        "overridden": True,
                        "params": 864.0,
                        "params_baseline": 864.0,
                        "perf_sensitivities": [
                            [0.0, 0.0],
                            [0.2, 2167603.1999999993],
                            [0.4, 4335206.4],
                            [0.6, 6502809.6],
                            [0.8, 8670412.8],
                            [0.9, 9754214.4],
                            [0.95, 10296115.2],
                            [0.99, 10729635.84],
                        ],
                        "sparsity": None,
                    },
                ],
                "optim_id": "8cc3f98370ad46f09103ee537872f41b",
                "params": 182491.20000000007,
                "params_baseline": 4210088.0,
                "sparsity": 0.85,
                "start_epoch": 1.0,
                "update_frequency": 0.825,
            },
            {
                "est_recovery": 0.6253260677145712,
                "flops_baseline": 569765352.0,
                "params_baseline": 4210088.0,
                "est_time_gain": None,
                "est_time_baseline": None,
                "params": 182491.20000000007,
                "created": "2020-10-29T15:12:30.726841",
                "modifier_id": "efbbd23e81c04129a916c5da5572df0e",
                "sparsity": 0.85,
                "balance_perf_loss": 1.0,
                "filter_min_perf_gain": 0.75,
                "est_time": None,
                "mask_type": "unstructured",
                "update_frequency": 0.825,
                "est_loss_sensitivity": 0.20195244026503392,
                "flops": 97905673.52,
                "est_perf_sensitivity": 19331324.442857143,
                "start_epoch": 1.0,
                "compression": 23.070087763135966,
                "nodes": [
                    {
                        "est_recovery": 1.0,
                        "est_time": None,
                        "params_baseline": 864.0,
                        "est_time_baseline": None,
                        "params": 864.0,
                        "compression": 1.0,
                        "flops_baseline": 10838016.0,
                        "node_id": "165",
                        "overridden": True,
                        "perf_sensitivities": [
                            (0.0, 0.0),
                            (0.2, 2167603.1999999993),
                            (0.4, 4335206.4),
                            (0.6, 6502809.6),
                            (0.8, 8670412.8),
                            (0.9, 9754214.4),
                            (0.95, 10296115.2),
                            (0.99, 10729635.84),
                        ],
                        "est_loss_sensitivity": 0.4556600749492645,
                        "flops": 10838016.0,
                        "flops_gain": 1.0,
                        "loss_sensitivities": [
                            (0.0, 0.0),
                            (0.2, 2.1940615828070636e-13),
                            (0.4, 3.5482580415524545e-12),
                            (0.6, 0.024138422682881355),
                            (0.8, 0.16036564111709595),
                            (0.9, 0.3132839798927307),
                            (0.95, 0.4556600749492645),
                            (0.99, 0.6454770565032959),
                        ],
                        "sparsity": None,
                        "est_perf_sensitivity": 10296115.2,
                        "est_time_gain": None,
                    }
                ],
                "modified": "2020-10-29T15:12:30.726846",
                "optim_id": "8cc3f98370ad46f09103ee537872f41b",
                "flops_gain": 5.819533552196129,
                "filter_min_sparsity": 0.4,
                "filter_min_recovery": -1.0,
                "end_epoch": 34.0,
            },
            None,
        ),
        (
            {
                "balance_perf_loss": 1.0,
                "compression": 23.070087763135966,
                "created": "2020-10-29T19:02:39.977360",
                "end_epoch": 34.0,
                "est_loss_sensitivity": 0.20195244026503392,
                "est_perf_sensitivity": 19331324.442857143,
                "est_recovery": 0.6253260677145712,
                "est_time": None,
                "est_time_baseline": None,
                "est_time_gain": None,
                "filter_min_perf_gain": 0.75,
                "filter_min_recovery": -1.0,
                "filter_min_sparsity": 0.4,
                "flops": 97905673.52,
                "flops_baseline": 569765352.0,
                "flops_gain": 5.819533552196129,
                "mask_type": "fail",
                "modified": "2020-10-29T19:02:40.818771",
                "modifier_id": "efbbd23e81c04129a916c5da5572df0e",
                "nodes": [
                    {
                        "compression": 1.0,
                        "est_loss_sensitivity": 0.4556600749492645,
                        "est_perf_sensitivity": 10296115.2,
                        "est_recovery": 1.0,
                        "est_time": None,
                        "est_time_baseline": None,
                        "est_time_gain": None,
                        "flops": 10838016.0,
                        "flops_baseline": 10838016.0,
                        "flops_gain": 1.0,
                        "loss_sensitivities": [
                            [0.0, 0.0],
                            [0.2, 2.1940615828070636e-13],
                            [0.4, 3.5482580415524545e-12],
                            [0.6, 0.024138422682881355],
                            [0.8, 0.16036564111709595],
                            [0.9, 0.3132839798927307],
                            [0.95, 0.4556600749492645],
                            [0.99, 0.6454770565032959],
                        ],
                        "node_id": "165",
                        "overridden": True,
                        "params": 864.0,
                        "params_baseline": 864.0,
                        "perf_sensitivities": [
                            [0.0, 0.0],
                            [0.2, 2167603.1999999993],
                            [0.4, 4335206.4],
                            [0.6, 6502809.6],
                            [0.8, 8670412.8],
                            [0.9, 9754214.4],
                            [0.95, 10296115.2],
                            [0.99, 10729635.84],
                        ],
                        "sparsity": None,
                    },
                ],
                "optim_id": "8cc3f98370ad46f09103ee537872f41b",
                "params": 182491.20000000007,
                "params_baseline": 4210088.0,
                "sparsity": 0.85,
                "start_epoch": 1.0,
                "update_frequency": 0.825,
            },
            None,
            ["mask_type"],
        ),
    ],
)
def test_project_optimization_modifier_pruning_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    modified = datetime.now()
    expected_input["created"] = created
    expected_input["modified"] = modified
    if expected_output:
        expected_output["created"] = created.isoformat()
        expected_output["modified"] = modified.isoformat()
    schema_tester(
        ProjectOptimizationModifierPruningSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        ({"node_id": None, "level": None}, {"node_id": None, "level": None}, None),
        (
            {"node_id": "node id", "level": "int8"},
            {"node_id": "node id", "level": "int8"},
            None,
        ),
        ({"node_id": "node id", "level": "fail"}, None, ["level"]),
    ],
)
def test_project_optimization_modifier_quantization_node_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        ProjectOptimizationModifierQuantizationNodeSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": None,
                "end_epoch": None,
                "level": None,
                "balance_perf_loss": 0,
                "filter_min_perf_gain": None,
                "filter_min_recovery": None,
                "nodes": [],
                "est_recovery": None,
                "est_perf_gain": None,
                "est_time": None,
                "est_time_baseline": None,
            },
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": None,
                "end_epoch": None,
                "level": None,
                "balance_perf_loss": 0,
                "filter_min_perf_gain": None,
                "filter_min_recovery": None,
                "nodes": [],
                "est_recovery": None,
                "est_perf_gain": None,
                "est_time": None,
                "est_time_baseline": None,
            },
            None,
        ),
        (
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": 1,
                "end_epoch": 50,
                "level": "int16",
                "balance_perf_loss": 1,
                "filter_min_perf_gain": 0.005,
                "filter_min_recovery": 1.5,
                "nodes": [
                    {"node_id": "node id", "level": "int16"},
                ],
                "est_recovery": 1,
                "est_perf_gain": 1,
                "est_time": 1,
                "est_time_baseline": 1,
            },
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": 1,
                "end_epoch": 50,
                "level": "int16",
                "balance_perf_loss": 1,
                "filter_min_perf_gain": 0.005,
                "filter_min_recovery": 1.5,
                "nodes": [
                    {"node_id": "node id", "level": "int16"},
                ],
                "est_recovery": 1,
                "est_perf_gain": 1,
                "est_time": 1,
                "est_time_baseline": 1,
            },
            None,
        ),
        (
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": 1,
                "end_epoch": 50,
                "level": "fail",
                "balance_perf_loss": 1,
                "filter_min_perf_gain": 0.005,
                "filter_min_recovery": 1.5,
                "nodes": [
                    {"node_id": "node id", "level": "int16"},
                ],
                "est_recovery": 1,
                "est_perf_gain": 1,
                "est_time": 1,
                "est_time_baseline": 1,
            },
            None,
            ["level"],
        ),
    ],
)
def test_project_optimization_modifier_quantization_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    created = datetime.now()
    modified = datetime.now()
    expected_input["created"] = created
    expected_input["modified"] = modified
    if expected_output:
        expected_output["created"] = created.isoformat()
        expected_output["modified"] = modified.isoformat()
    schema_tester(
        ProjectOptimizationModifierQuantizationSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {
                "clazz": "set",
                "start_epoch": -1,
                "end_epoch": 100,
                "init_lr": 0.005,
                "args": {},
            },
            {
                "clazz": "set",
                "start_epoch": -1,
                "end_epoch": 100,
                "init_lr": 0.005,
                "args": {},
            },
            None,
        ),
        (
            {
                "clazz": "step",
                "start_epoch": -1,
                "end_epoch": 100,
                "init_lr": 0.005,
                "args": {"gamma": 0.25, "step_size": 5},
            },
            {
                "clazz": "step",
                "start_epoch": -1,
                "end_epoch": 100,
                "init_lr": 0.005,
                "args": {"gamma": 0.25, "step_size": 5},
            },
            None,
        ),
        (
            {
                "clazz": "fail",
                "start_epoch": -1,
                "end_epoch": 100,
                "init_lr": 0.005,
                "args": {"gamma": 0.25, "step_size": 5},
            },
            None,
            ["clazz"],
        ),
    ],
)
def test_project_optimization_modifier_lr_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        ProjectOptimizationModifierLRSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize("expected_input,expected_output", [({}, {})])
def test_project_optimization_modifier_lr_set_args_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectOptimizationModifierLRSetArgsSchema, expected_input, expected_output
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "step_size": 1,
            },
            {"step_size": 1, "gamma": 0.1},
        ),
        (
            {"step_size": 1, "gamma": 0.01},
            {"step_size": 1, "gamma": 0.01},
        ),
    ],
)
def test_project_optimization_modifier_lr_step_args_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectOptimizationModifierLRStepArgsSchema, expected_input, expected_output
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "milestones": [1],
            },
            {"milestones": [1], "gamma": 0.1},
        ),
        (
            {"milestones": [1, 2], "gamma": 0.01},
            {"milestones": [1, 2], "gamma": 0.01},
        ),
    ],
)
def test_project_optimization_modifier_lr_multistep_args_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectOptimizationModifierLRMultiStepArgsSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {},
            {"gamma": 0.1},
        ),
        (
            {"gamma": 0.01},
            {"gamma": 0.01},
        ),
    ],
)
def test_project_optimization_modifier_lr_exponential_args_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectOptimizationModifierLRExponentialArgsSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": None,
                "end_epoch": None,
                "init_lr": None,
                "final_lr": None,
                "lr_mods": [],
            },
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": None,
                "end_epoch": None,
                "init_lr": None,
                "final_lr": None,
                "lr_mods": [],
            },
        ),
        (
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": 1,
                "end_epoch": 100,
                "init_lr": 0.005,
                "final_lr": 0.00005,
                "lr_mods": [
                    {
                        "clazz": "multi_step",
                        "start_epoch": -1,
                        "end_epoch": 50,
                        "init_lr": 0.005,
                        "args": {"milestones": [1, 20], "gamma": 0.01},
                    },
                    {
                        "clazz": "exponential",
                        "start_epoch": 50,
                        "end_epoch": 100,
                        "init_lr": 0.005,
                        "args": {"gamma": 0.01},
                    },
                ],
            },
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": 1,
                "end_epoch": 100,
                "init_lr": 0.005,
                "final_lr": 0.00005,
                "lr_mods": [
                    {
                        "clazz": "multi_step",
                        "start_epoch": -1,
                        "end_epoch": 50,
                        "init_lr": 0.005,
                        "args": {"milestones": [1, 20], "gamma": 0.01},
                    },
                    {
                        "clazz": "exponential",
                        "start_epoch": 50,
                        "end_epoch": 100,
                        "init_lr": 0.005,
                        "args": {"gamma": 0.01},
                    },
                ],
            },
        ),
    ],
)
def test_project_optimization_modifier_lr_schedule_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    created = datetime.now()
    modified = datetime.now()
    expected_input["created"] = created
    expected_input["modified"] = modified
    if expected_output:
        expected_output["created"] = created.isoformat()
        expected_output["modified"] = modified.isoformat()
    schema_tester(
        ProjectOptimizationModifierLRScheduleSchema, expected_input, expected_output
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "node_id": None,
                "trainable": None,
            },
            {
                "node_id": None,
                "trainable": None,
            },
        ),
        (
            {
                "node_id": "node id",
                "trainable": True,
            },
            {
                "node_id": "node id",
                "trainable": True,
            },
        ),
    ],
)
def test_project_optimization_modifier_trainable_node_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ProjectOptimizationModifierTrainableNodeSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": None,
                "end_epoch": None,
                "nodes": [],
            },
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": None,
                "end_epoch": None,
                "nodes": [],
            },
        ),
        (
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": 1,
                "end_epoch": 100,
                "nodes": [
                    {
                        "node_id": "node id",
                        "trainable": True,
                    },
                    {
                        "node_id": "node id 2",
                        "trainable": False,
                    },
                ],
            },
            {
                "modifier_id": "modifier id",
                "optim_id": "optim id",
                "start_epoch": 1,
                "end_epoch": 100,
                "nodes": [
                    {
                        "node_id": "node id",
                        "trainable": True,
                    },
                    {
                        "node_id": "node id 2",
                        "trainable": False,
                    },
                ],
            },
        ),
    ],
)
def test_project_optimization_modifier_trainable_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    created = datetime.now()
    modified = datetime.now()
    expected_input["created"] = created
    expected_input["modified"] = modified
    if expected_output:
        expected_output["created"] = created.isoformat()
        expected_output["modified"] = modified.isoformat()
    schema_tester(
        ProjectOptimizationModifierTrainableSchema, expected_input, expected_output
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "optim_id": "optim id",
                "project_id": "project id",
                "name": None,
                "profile_perf_id": None,
                "profile_loss_id": None,
                "start_epoch": 0,
                "end_epoch": 100,
                "pruning_modifiers": [],
                "quantization_modifiers": [],
                "lr_schedule_modifiers": [],
                "trainable_modifiers": [],
            },
            {
                "optim_id": "optim id",
                "project_id": "project id",
                "name": None,
                "profile_perf_id": None,
                "profile_loss_id": None,
                "start_epoch": 0,
                "end_epoch": 100,
                "pruning_modifiers": [],
                "quantization_modifiers": [],
                "lr_schedule_modifiers": [],
                "trainable_modifiers": [],
            },
        ),
        (
            {
                "optim_id": "optim id",
                "project_id": "project id",
                "name": "name",
                "profile_perf_id": "profile perf id",
                "profile_loss_id": "profile loss id",
                "start_epoch": 0,
                "end_epoch": 100,
                "pruning_modifiers": [
                    {
                        "modifier_id": "modifier id",
                        "optim_id": "optim id",
                        "start_epoch": 1,
                        "end_epoch": 50,
                        "update_frequency": 1,
                        "mask_type": "block_2",
                        "sparsity": 0.5,
                        "balance_perf_loss": 1,
                        "filter_min_sparsity": 0.4,
                        "filter_min_perf_gain": 0.005,
                        "filter_min_recovery": 1.5,
                        "flops": 200,
                        "flops_baseline": 200,
                        "params": 20,
                        "params_baseline": 20,
                        "nodes": [
                            {
                                "compression": 1.0,
                                "est_loss_sensitivity": 0.4556600749492645,
                                "est_perf_sensitivity": 10296115.2,
                                "est_recovery": 1.0,
                                "est_time": None,
                                "est_time_baseline": None,
                                "est_time_gain": None,
                                "flops": 10838016.0,
                                "flops_baseline": 10838016.0,
                                "flops_gain": 1.0,
                                "loss_sensitivities": [
                                    [0.0, 0.0],
                                    [0.2, 2.1940615828070636e-13],
                                    [0.4, 3.5482580415524545e-12],
                                    [0.6, 0.024138422682881355],
                                    [0.8, 0.16036564111709595],
                                    [0.9, 0.3132839798927307],
                                    [0.95, 0.4556600749492645],
                                    [0.99, 0.6454770565032959],
                                ],
                                "node_id": "165",
                                "overridden": True,
                                "params": 864.0,
                                "params_baseline": 864.0,
                                "perf_sensitivities": [
                                    [0.0, 0.0],
                                    [0.2, 2167603.1999999993],
                                    [0.4, 4335206.4],
                                    [0.6, 6502809.6],
                                    [0.8, 8670412.8],
                                    [0.9, 9754214.4],
                                    [0.95, 10296115.2],
                                    [0.99, 10729635.84],
                                ],
                                "sparsity": None,
                            },
                        ],
                        "est_recovery": 1,
                        "est_perf_gain": 1,
                        "est_time": 1,
                        "est_time_gain": 1,
                        "est_time_baseline": 1,
                        "flops_gain": 5.819533552196129,
                        "est_loss_sensitivity": 551.319094694324,
                        "est_perf_sensitivity": 19331324.442857143,
                        "compression": 23.070087763135966,
                    },
                ],
                "quantization_modifiers": [
                    {
                        "modifier_id": "modifier id",
                        "optim_id": "optim id",
                        "start_epoch": 1,
                        "end_epoch": 50,
                        "level": "int16",
                        "balance_perf_loss": 1,
                        "filter_min_perf_gain": 0.005,
                        "filter_min_recovery": 1.5,
                        "nodes": [
                            {"node_id": "node id", "level": "int16"},
                        ],
                        "est_recovery": 1,
                        "est_perf_gain": 1,
                        "est_time": 1,
                        "est_time_baseline": 1,
                    },
                ],
                "lr_schedule_modifiers": [
                    {
                        "modifier_id": "modifier id",
                        "optim_id": "optim id",
                        "start_epoch": 1,
                        "end_epoch": 100,
                        "init_lr": 0.005,
                        "final_lr": 0.00005,
                        "lr_mods": [
                            {
                                "clazz": "multi_step",
                                "start_epoch": -1,
                                "end_epoch": 50,
                                "init_lr": 0.005,
                                "args": {"milestones": [1, 20], "gamma": 0.01},
                            },
                            {
                                "clazz": "exponential",
                                "start_epoch": 50,
                                "end_epoch": 100,
                                "init_lr": 0.005,
                                "args": {"gamma": 0.01},
                            },
                        ],
                    },
                ],
                "trainable_modifiers": [
                    {
                        "modifier_id": "modifier id",
                        "optim_id": "optim id",
                        "start_epoch": 1,
                        "end_epoch": 100,
                        "nodes": [
                            {
                                "node_id": "node id",
                                "trainable": True,
                            },
                            {
                                "node_id": "node id 2",
                                "trainable": False,
                            },
                        ],
                    },
                ],
            },
            {
                "profile_loss_id": "profile loss id",
                "start_epoch": 0.0,
                "optim_id": "optim id",
                "modified": "2020-10-29T15:08:07.714614",
                "end_epoch": 100.0,
                "name": "name",
                "quantization_modifiers": [
                    {
                        "nodes": [{"level": "int16", "node_id": "node id"}],
                        "est_time": 1.0,
                        "start_epoch": 1.0,
                        "level": "int16",
                        "modifier_id": "modifier id",
                        "optim_id": "optim id",
                        "modified": "2020-10-29T15:08:07.714614",
                        "end_epoch": 50.0,
                        "filter_min_perf_gain": 0.005,
                        "filter_min_recovery": 1.5,
                        "est_time_baseline": 1.0,
                        "est_recovery": 1.0,
                        "est_perf_gain": 1.0,
                        "balance_perf_loss": 1.0,
                        "created": "2020-10-29T15:08:07.714610",
                    }
                ],
                "lr_schedule_modifiers": [
                    {
                        "start_epoch": 1.0,
                        "modifier_id": "modifier id",
                        "optim_id": "optim id",
                        "modified": "2020-10-29T15:08:07.714614",
                        "end_epoch": 100.0,
                        "init_lr": 0.005,
                        "lr_mods": [
                            {
                                "args": {"milestones": [1, 20], "gamma": 0.01},
                                "start_epoch": -1.0,
                                "end_epoch": 50.0,
                                "clazz": "multi_step",
                                "init_lr": 0.005,
                            },
                            {
                                "args": {"gamma": 0.01},
                                "start_epoch": 50.0,
                                "end_epoch": 100.0,
                                "clazz": "exponential",
                                "init_lr": 0.005,
                            },
                        ],
                        "final_lr": 5e-05,
                        "created": "2020-10-29T15:08:07.714610",
                    }
                ],
                "trainable_modifiers": [
                    {
                        "nodes": [
                            {"trainable": True, "node_id": "node id"},
                            {"trainable": False, "node_id": "node id 2"},
                        ],
                        "start_epoch": 1.0,
                        "modifier_id": "modifier id",
                        "optim_id": "optim id",
                        "modified": "2020-10-29T15:08:07.714614",
                        "end_epoch": 100.0,
                        "created": "2020-10-29T15:08:07.714610",
                    }
                ],
                "profile_perf_id": "profile perf id",
                "project_id": "project id",
                "pruning_modifiers": [
                    {
                        "nodes": [
                            {
                                "overridden": True,
                                "compression": 1.0,
                                "flops_gain": 1.0,
                                "est_time_gain": None,
                                "sparsity": None,
                                "node_id": "165",
                                "perf_sensitivities": [
                                    (0.0, 0.0),
                                    (0.2, 2167603.1999999993),
                                    (0.4, 4335206.4),
                                    (0.6, 6502809.6),
                                    (0.8, 8670412.8),
                                    (0.9, 9754214.4),
                                    (0.95, 10296115.2),
                                    (0.99, 10729635.84),
                                ],
                                "params": 864.0,
                                "est_loss_sensitivity": 0.4556600749492645,
                                "flops_baseline": 10838016.0,
                                "loss_sensitivities": [
                                    (0.0, 0.0),
                                    (0.2, 2.1940615828070636e-13),
                                    (0.4, 3.5482580415524545e-12),
                                    (0.6, 0.024138422682881355),
                                    (0.8, 0.16036564111709595),
                                    (0.9, 0.3132839798927307),
                                    (0.95, 0.4556600749492645),
                                    (0.99, 0.6454770565032959),
                                ],
                                "est_time_baseline": None,
                                "est_recovery": 1.0,
                                "flops": 10838016.0,
                                "params_baseline": 864.0,
                                "est_time": None,
                                "est_perf_sensitivity": 10296115.2,
                            }
                        ],
                        "start_epoch": 1.0,
                        "filter_min_recovery": 1.5,
                        "est_time_baseline": 1.0,
                        "params": 20.0,
                        "flops": 200.0,
                        "params_baseline": 20.0,
                        "compression": 23.070087763135966,
                        "flops_gain": 5.819533552196129,
                        "end_epoch": 50.0,
                        "filter_min_perf_gain": 0.005,
                        "flops_baseline": 200.0,
                        "sparsity": 0.5,
                        "mask_type": "block_2",
                        "est_time_gain": 1.0,
                        "modifier_id": "modifier id",
                        "est_loss_sensitivity": 551.319094694324,
                        "est_recovery": 1.0,
                        "created": "2020-10-29T15:08:07.714610",
                        "est_perf_sensitivity": 19331324.442857143,
                        "balance_perf_loss": 1.0,
                        "filter_min_sparsity": 0.4,
                        "optim_id": "optim id",
                        "modified": "2020-10-29T15:08:07.714614",
                        "update_frequency": 1.0,
                        "est_time": 1.0,
                    }
                ],
                "created": "2020-10-29T15:08:07.714610",
            },
        ),
    ],
)
def test_project_optimization_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    created = datetime.now()
    modified = datetime.now()
    expected_input["created"] = created
    expected_input["modified"] = modified
    expected_output["created"] = created.isoformat()
    expected_output["modified"] = modified.isoformat()
    modifier_keys = [
        "pruning_modifiers",
        "quantization_modifiers",
        "lr_schedule_modifiers",
        "trainable_modifiers",
    ]
    for mods in modifier_keys:
        for mod in expected_input[mods]:
            mod["created"] = created
            mod["modified"] = modified

        for mod in expected_output[mods]:
            mod["created"] = created.isoformat()
            mod["modified"] = modified.isoformat()

    schema_tester(ProjectOptimizationSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({}, {"profile_perf_id": None, "profile_loss_id": None}),
        (
            {"profile_perf_id": "perf id", "profile_loss_id": "loss id"},
            {"profile_perf_id": "perf id", "profile_loss_id": "loss id"},
        ),
    ],
)
def test_get_project_optimization_best_estimated_results_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        GetProjectOptimizationBestEstimatedResultsSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {},
            {
                "name": "",
                "add_pruning": True,
                "add_quantization": False,
                "add_lr_schedule": True,
                "add_trainable": False,
                "profile_perf_id": None,
                "profile_loss_id": None,
            },
        ),
        (
            {
                "name": "name",
                "add_pruning": False,
                "add_quantization": True,
                "add_lr_schedule": False,
                "add_trainable": True,
                "profile_perf_id": "perf id",
                "profile_loss_id": "loss id",
            },
            {
                "name": "name",
                "add_pruning": False,
                "add_quantization": True,
                "add_lr_schedule": False,
                "add_trainable": True,
                "profile_perf_id": "perf id",
                "profile_loss_id": "loss id",
            },
        ),
    ],
)
def test_create_project_optimization_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(CreateProjectOptimizationSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({}, {}),
        (
            {
                "name": "name",
                "profile_perf_id": "perf id",
                "profile_loss_id": "loss id",
                "start_epoch": 1,
                "end_epoch": 100,
            },
            {
                "name": "name",
                "profile_perf_id": "perf id",
                "profile_loss_id": "loss id",
                "start_epoch": 1,
                "end_epoch": 100,
            },
        ),
    ],
)
def test_update_project_optimization_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(UpdateProjectOptimizationSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({}, {}),
        (
            {
                "start_epoch": 1,
                "end_epoch": 100,
                "update_frequency": 1,
                "sparsity": 0.5,
                "balance_perf_loss": 0,
                "filter_min_sparsity": 0.4,
                "filter_min_perf_gain": 2,
                "filter_min_recovery": 1,
                "nodes": [
                    {
                        "node_id": "node id",
                        "sparsity": 0.75,
                    },
                ],
            },
            {
                "start_epoch": 1,
                "end_epoch": 100,
                "update_frequency": 1,
                "sparsity": 0.5,
                "balance_perf_loss": 0,
                "filter_min_sparsity": 0.4,
                "filter_min_perf_gain": 2,
                "filter_min_recovery": 1,
                "nodes": [
                    {
                        "node_id": "node id",
                        "sparsity": 0.75,
                    },
                ],
            },
        ),
    ],
)
def test_create_update_project_optimization_pruning_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        CreateUpdateProjectOptimizationModifiersPruningSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        ({}, {}, None),
        (
            {
                "start_epoch": 1,
                "end_epoch": 100,
                "level": "int8",
                "balance_perf_loss": 0,
                "filter_min_perf_gain": 2,
                "filter_min_recovery": 1,
                "nodes": [
                    {"node_id": "node id", "level": "int16"},
                ],
            },
            {
                "start_epoch": 1,
                "end_epoch": 100,
                "level": "int8",
                "balance_perf_loss": 0,
                "filter_min_perf_gain": 2.0,
                "filter_min_recovery": 1.0,
                "nodes": [
                    {"node_id": "node id", "level": "int16"},
                ],
            },
            None,
        ),
        (
            {
                "start_epoch": 1,
                "end_epoch": 100,
                "level": "fail",
                "balance_perf_loss": 0,
                "filter_min_perf_gain": 2,
                "filter_min_recovery": 1,
                "nodes": [
                    {"node_id": "node id", "level": "int16"},
                ],
            },
            None,
            ["level"],
        ),
    ],
)
def test_create_update_project_optimization_quantization_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        CreateUpdateProjectOptimizationModifiersQuantizationSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({}, {}),
        (
            {
                "lr_mods": [
                    {
                        "clazz": "multi_step",
                        "start_epoch": -1,
                        "end_epoch": 50,
                        "init_lr": 0.005,
                        "args": {"milestones": [1, 20], "gamma": 0.01},
                    },
                    {
                        "clazz": "exponential",
                        "start_epoch": 50,
                        "end_epoch": 100,
                        "init_lr": 0.005,
                        "args": {"gamma": 0.01},
                    },
                ]
            },
            {
                "lr_mods": [
                    {
                        "clazz": "multi_step",
                        "start_epoch": -1,
                        "end_epoch": 50,
                        "init_lr": 0.005,
                        "args": {"milestones": [1, 20], "gamma": 0.01},
                    },
                    {
                        "clazz": "exponential",
                        "start_epoch": 50,
                        "end_epoch": 100,
                        "init_lr": 0.005,
                        "args": {"gamma": 0.01},
                    },
                ]
            },
        ),
    ],
)
def test_create_update_project_optimization_lr_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        CreateUpdateProjectOptimizationModifiersLRScheduleSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({}, {}),
        (
            {
                "start_epoch": 1,
                "end_epoch": 100,
                "nodes": [
                    {
                        "node_id": "node id",
                        "trainable": False,
                    },
                ],
            },
            {
                "start_epoch": 1,
                "end_epoch": 100,
                "nodes": [
                    {
                        "node_id": "node id",
                        "trainable": False,
                    },
                ],
            },
        ),
    ],
)
def test_create_update_project_optimization_trainable_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        CreateUpdateProjectOptimizationModifiersTrainableSchema,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        ({}, {"page": 1, "page_length": 20}, None),
        (
            {"page": 2, "page_length": 21},
            {"page": 2, "page_length": 21},
            None,
        ),
        (
            {"page": 0, "page_length": 0},
            None,
            ["page", "page_length"],
        ),
    ],
)
def test_search_project_optimization_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        SearchProjectOptimizationsSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        ({"frameworks": ["pytorch"]}, {"frameworks": ["pytorch"]}, None),
        (
            {"frameworks": ["pytorch", "tensorflow"]},
            {"frameworks": ["pytorch", "tensorflow"]},
            None,
        ),
        ({"frameworks": "fail"}, None, ["frameworks"]),
    ],
)
def test_response_project_optimization_frameworks_available_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        ResponseProjectOptimizationFrameworksAvailableSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {"framework": "pytorch", "samples": []},
            {"framework": "pytorch", "samples": []},
            None,
        ),
        (
            {"framework": "tensorflow", "samples": ["sample line"]},
            {"framework": "tensorflow", "samples": ["sample line"]},
            None,
        ),
        ({"framework": "fail", "samples": []}, None, ["framework"]),
    ],
)
def test_response_project_optimization_frameworks_available_samples_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        ResponseProjectOptimizationFrameworksAvailableSamplesSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        ({"modifiers": ["pruning"]}, {"modifiers": ["pruning"]}, None),
        (
            {"modifiers": ["trainable", "quantization"]},
            {"modifiers": ["trainable", "quantization"]},
            None,
        ),
        ({"modifiers": "fail"}, None, ["modifiers"]),
    ],
)
def test_response_project_optimization_modifiers_available_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        ResponseProjectOptimizationModifiersAvailable,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "est_time_baseline": None,
                "est_time_gain": None,
                "flops_gain": None,
                "est_loss_sensitivity": None,
                "params_baseline": None,
                "est_perf_sensitivity": None,
                "est_time": None,
                "flops_baseline": None,
                "est_recovery": None,
                "params": None,
                "flops": None,
                "compression": None,
            },
            {
                "est_time_baseline": None,
                "est_time_gain": None,
                "flops_gain": None,
                "est_loss_sensitivity": None,
                "params_baseline": None,
                "est_perf_sensitivity": None,
                "est_time": None,
                "flops_baseline": None,
                "est_recovery": None,
                "params": None,
                "flops": None,
                "compression": None,
            },
        ),
        (
            {
                "compression": 23.070087763135966,
                "est_loss_sensitivity": 551.319094694324,
                "est_perf_sensitivity": 19331324.442857143,
                "est_recovery": 1.0,
                "est_time": 23.070087763135966,
                "est_time_baseline": 23.070087763135966,
                "est_time_gain": 23.070087763135966,
                "flops": 97905673.52,
                "flops_baseline": 569765352.0,
                "flops_gain": 5.819533552196129,
                "params": 182491.20000000007,
                "params_baseline": 4210088.0,
            },
            {
                "compression": 23.070087763135966,
                "est_loss_sensitivity": 551.319094694324,
                "est_perf_sensitivity": 19331324.442857143,
                "est_recovery": 1.0,
                "est_time": 23.070087763135966,
                "est_time_baseline": 23.070087763135966,
                "est_time_gain": 23.070087763135966,
                "flops": 97905673.52,
                "flops_baseline": 569765352.0,
                "flops_gain": 5.819533552196129,
                "params": 182491.20000000007,
                "params_baseline": 4210088.0,
            },
        ),
    ],
)
def test_response_project_optimization_best_estimated(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ResponseProjectOptimizationModifiersBestEstimated,
        expected_input,
        expected_output,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "optim": {
                    "optim_id": "optim id",
                    "project_id": "project id",
                    "name": "name",
                    "profile_perf_id": "profile perf id",
                    "profile_loss_id": "profile loss id",
                    "start_epoch": 0,
                    "end_epoch": 100,
                    "pruning_modifiers": [],
                    "quantization_modifiers": [],
                    "lr_schedule_modifiers": [],
                    "trainable_modifiers": [],
                }
            },
            {
                "optim": {
                    "optim_id": "optim id",
                    "project_id": "project id",
                    "name": "name",
                    "profile_perf_id": "profile perf id",
                    "profile_loss_id": "profile loss id",
                    "start_epoch": 0,
                    "end_epoch": 100,
                    "pruning_modifiers": [],
                    "quantization_modifiers": [],
                    "lr_schedule_modifiers": [],
                    "trainable_modifiers": [],
                }
            },
        ),
    ],
)
def test_response_project_optimization_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    created = datetime.now()
    modified = datetime.now()
    expected_input["optim"]["created"] = created
    expected_input["optim"]["modified"] = modified
    expected_output["optim"]["created"] = created.isoformat()
    expected_output["optim"]["modified"] = modified.isoformat()
    modifier_keys = [
        "pruning_modifiers",
        "quantization_modifiers",
        "lr_schedule_modifiers",
        "trainable_modifiers",
    ]
    for mods in modifier_keys:
        for mod in expected_input["optim"][mods]:
            mod["created"] = created
            mod["modified"] = modified

        for mod in expected_output["optim"][mods]:
            mod["created"] = created.isoformat()
            mod["modified"] = modified.isoformat()

    schema_tester(ResponseProjectOptimizationSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"project_id": "project id", "optim_id": "optim id"},
            {"success": True, "project_id": "project id", "optim_id": "optim id"},
        ),
        (
            {"success": False, "project_id": "project id", "optim_id": "optim id"},
            {"success": False, "project_id": "project id", "optim_id": "optim id"},
        ),
    ],
)
def test_response_project_optimization_deleted_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ResponseProjectOptimizationDeletedSchema, expected_input, expected_output
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "project_id": "project id",
                "optim_id": "optim id",
                "modifer_id": "modifer id",
            },
            {
                "success": True,
                "project_id": "project id",
                "optim_id": "optim id",
                "modifer_id": "modifer id",
            },
        ),
        (
            {
                "success": False,
                "project_id": "project id",
                "optim_id": "optim id",
                "modifer_id": "modifer id",
            },
            {
                "success": False,
                "project_id": "project id",
                "optim_id": "optim id",
                "modifer_id": "modifer id",
            },
        ),
    ],
)
def test_response_project_optimization_modifier_deleted_schema(
    expected_input: Dict,
    expected_output: Dict,
):
    schema_tester(
        ResponseProjectOptimizationModifierDeletedSchema,
        expected_input,
        expected_output,
    )
