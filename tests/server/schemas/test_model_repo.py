from typing import Dict, List, Union

import pytest
from neuralmagicML.server.schemas.model_repo import (
    ModelRepoModelPerfSchema,
    ModelRepoModelMetricSchema,
    ModelRepoModelSchema,
    ModelRepoDomainSchema,
    ModelRepoArchitectureSchema,
    ModelRepoDatasetSchema,
    ModelRepoModelDescSchema,
    SearchModelRepoModels,
    ResponseModelRepoModels,
)

from tests.server.helper import schema_tester


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"seconds_per_batch": 1, "batch_size": 3.0, "cpu_core_count": 1.0},
            {"seconds_per_batch": 1.0, "batch_size": 3, "cpu_core_count": 1},
        )
    ],
)
def test_model_repo_model_perf_schema(
    expected_input: Dict, expected_output: Dict,
):
    schema_tester(ModelRepoModelPerfSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output,expect_validation_error",
    [
        (
            {"value": 1.0, "label": "label", "display_type": "number"},
            {"value": 1.0, "label": "label", "display_type": "number"},
            None,
        ),
        (
            {"value": 1.0, "label": "label", "display_type": "fail"},
            None,
            ["display_type"],
        ),
    ],
)
def test_model_repo_model_metric_schema(
    expected_input: Dict,
    expected_output: Dict,
    expect_validation_error: Union[List[str], None],
):
    schema_tester(
        ModelRepoModelMetricSchema,
        expected_input,
        expected_output,
        expect_validation_error,
    )


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "display_name": "name",
                "display_summary": "summary",
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "resnet",
                "sub_architecture": "50",
                "dataset": "imagenet",
                "framework": "pytorch",
                "desc": "base",
                "latency": {
                    "seconds_per_batch": 1.0,
                    "batch_size": 3,
                    "cpu_core_count": 1,
                },
                "throughput": {
                    "seconds_per_batch": 1.0,
                    "batch_size": 3,
                    "cpu_core_count": 1,
                },
                "metrics": [{"value": 1.0, "label": "label", "display_type": "number"}],
            },
            {
                "display_name": "name",
                "display_summary": "summary",
                "domain": "cv",
                "sub_domain": "classification",
                "architecture": "resnet",
                "sub_architecture": "50",
                "dataset": "imagenet",
                "framework": "pytorch",
                "desc": "base",
                "latency": {
                    "seconds_per_batch": 1.0,
                    "batch_size": 3,
                    "cpu_core_count": 1,
                },
                "throughput": {
                    "seconds_per_batch": 1.0,
                    "batch_size": 3,
                    "cpu_core_count": 1,
                },
                "metrics": [{"value": 1.0, "label": "label", "display_type": "number"}],
            },
        )
    ],
)
def test_model_repo_model_schema(
    expected_input: Dict, expected_output: Dict,
):
    schema_tester(ModelRepoModelSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"domain": "cv", "sub_domain": "classification"},
            {"display": None, "domain": "cv", "sub_domain": "classification"},
        ),
        (
            {
                "display": "cv-classification",
                "domain": "cv",
                "sub_domain": "classification",
            },
            {
                "display": "cv-classification",
                "domain": "cv",
                "sub_domain": "classification",
            },
        ),
    ],
)
def test_model_repo_domain_schema(
    expected_input: Dict, expected_output: Dict,
):
    schema_tester(ModelRepoDomainSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {"architecture": "resnet", "sub_architecture": "50"},
            {"display": None, "architecture": "resnet", "sub_architecture": "50"},
        ),
        (
            {
                "display": "resnet-50",
                "architecture": "resnet",
                "sub_architecture": "50",
            },
            {
                "display": "resnet-50",
                "architecture": "resnet",
                "sub_architecture": "50",
            },
        ),
    ],
)
def test_model_repo_architecture_schema(
    expected_input: Dict, expected_output: Dict,
):
    schema_tester(ModelRepoArchitectureSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({"dataset": "imagenet"}, {"display": None, "dataset": "imagenet"}),
        (
            {"display": "Imagenet", "dataset": "imagenet"},
            {"display": "Imagenet", "dataset": "imagenet"},
        ),
    ],
)
def test_model_repo_dataset_schema(
    expected_input: Dict, expected_output: Dict,
):
    schema_tester(ModelRepoDatasetSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        ({"desc": "base"}, {"display": None, "desc": "base"}),
        ({"display": "Base", "desc": "base"}, {"display": "Base", "desc": "base"},),
    ],
)
def test_model_repo_desc_schema(
    expected_input: Dict, expected_output: Dict,
):
    schema_tester(ModelRepoModelDescSchema, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {},
            {
                "filter_domains": None,
                "filter_architectures": None,
                "filter_datasets": None,
                "filter_model_descs": None,
            },
        ),
        (
            {
                "filter_domains": [
                    {
                        "display": "cv-classification",
                        "domain": "cv",
                        "sub_domain": "classification",
                    }
                ],
                "filter_architectures": [
                    {
                        "display": "resnet-50",
                        "architecture": "resnet",
                        "sub_architecture": "50",
                    }
                ],
                "filter_datasets": [{"dataset": "imagenet", "display": "Imagenet"}],
                "filter_model_descs": [{"display": "Base", "desc": "base"}],
            },
            {
                "filter_domains": [
                    {
                        "display": "cv-classification",
                        "domain": "cv",
                        "sub_domain": "classification",
                    }
                ],
                "filter_architectures": [
                    {
                        "display": "resnet-50",
                        "architecture": "resnet",
                        "sub_architecture": "50",
                    }
                ],
                "filter_datasets": [{"dataset": "imagenet", "display": "Imagenet"}],
                "filter_model_descs": [{"display": "Base", "desc": "base"}],
            },
        ),
    ],
)
def test_search_model_repo_schema(
    expected_input: Dict, expected_output: Dict,
):
    schema_tester(SearchModelRepoModels, expected_input, expected_output)


@pytest.mark.parametrize(
    "expected_input,expected_output",
    [
        (
            {
                "models": [],
                "domains": [
                    {
                        "display": "cv-classification",
                        "domain": "cv",
                        "sub_domain": "classification",
                    }
                ],
                "architectures": [
                    {
                        "display": "resnet-50",
                        "architecture": "resnet",
                        "sub_architecture": "50",
                    }
                ],
                "datasets": [{"dataset": "imagenet", "display": "Imagenet"}],
                "model_descs": [{"display": "Base", "desc": "base"}],
            },
            {
                "models": [],
                "domains": [
                    {
                        "display": "cv-classification",
                        "domain": "cv",
                        "sub_domain": "classification",
                    }
                ],
                "architectures": [
                    {
                        "display": "resnet-50",
                        "architecture": "resnet",
                        "sub_architecture": "50",
                    }
                ],
                "datasets": [{"dataset": "imagenet", "display": "Imagenet"}],
                "model_descs": [{"display": "Base", "desc": "base"}],
            },
        ),
        (
            {
                "models": [
                    {
                        "display_name": "name",
                        "display_summary": "summary",
                        "domain": "cv",
                        "sub_domain": "classification",
                        "architecture": "resnet",
                        "sub_architecture": "50",
                        "dataset": "imagenet",
                        "framework": "pytorch",
                        "desc": "base",
                        "latency": {
                            "seconds_per_batch": 1.0,
                            "batch_size": 3,
                            "cpu_core_count": 1,
                        },
                        "throughput": {
                            "seconds_per_batch": 1.0,
                            "batch_size": 3,
                            "cpu_core_count": 1,
                        },
                        "metrics": [
                            {"value": 1.0, "label": "label", "display_type": "number"}
                        ],
                    },
                ],
                "domains": [
                    {
                        "display": "cv-classification",
                        "domain": "cv",
                        "sub_domain": "classification",
                    }
                ],
                "architectures": [
                    {
                        "display": "resnet-50",
                        "architecture": "resnet",
                        "sub_architecture": "50",
                    }
                ],
                "datasets": [{"dataset": "imagenet", "display": "Imagenet"}],
                "model_descs": [{"display": "Base", "desc": "base"}],
            },
            {
                "models": [
                    {
                        "display_name": "name",
                        "display_summary": "summary",
                        "domain": "cv",
                        "sub_domain": "classification",
                        "architecture": "resnet",
                        "sub_architecture": "50",
                        "dataset": "imagenet",
                        "framework": "pytorch",
                        "desc": "base",
                        "latency": {
                            "seconds_per_batch": 1.0,
                            "batch_size": 3,
                            "cpu_core_count": 1,
                        },
                        "throughput": {
                            "seconds_per_batch": 1.0,
                            "batch_size": 3,
                            "cpu_core_count": 1,
                        },
                        "metrics": [
                            {"value": 1.0, "label": "label", "display_type": "number"}
                        ],
                    },
                ],
                "domains": [
                    {
                        "display": "cv-classification",
                        "domain": "cv",
                        "sub_domain": "classification",
                    }
                ],
                "architectures": [
                    {
                        "display": "resnet-50",
                        "architecture": "resnet",
                        "sub_architecture": "50",
                    }
                ],
                "datasets": [{"dataset": "imagenet", "display": "Imagenet"}],
                "model_descs": [{"display": "Base", "desc": "base"}],
            },
        ),
    ],
)
def test_response_model_repo_schema(
    expected_input: Dict, expected_output: Dict,
):
    schema_tester(ResponseModelRepoModels, expected_input, expected_output)
