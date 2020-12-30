import csv
import json
import os
import tempfile
from collections import OrderedDict
from typing import Dict, List, NamedTuple, Tuple, Union

import pytest
from sparseml.onnx.optim.analyzer_model import ModelAnalyzer, NodeAnalyzer
from sparseml.onnx.optim.sensitivity_info import (
    SensitivityModelInfo,
    SensitivityNodeInfo,
)
from sparseml.optim.sensitivity import (
    PruningLossSensitivityAnalysis,
    PruningPerfSensitivityAnalysis,
    PruningSensitivityResult,
)

from tests.onnx.helpers import GENERATE_TEST_FILES, onnx_repo_models
from tests.onnx.optim.test_sensitivity_ks import (
    OnnxModelAnalysisFixture,
    onnx_models_with_analysis,
)

RELATIVE_PATH = os.path.dirname(os.path.realpath(__file__))
SENSITIVITY_INFO_DIR = os.path.join(RELATIVE_PATH, "test_sensitivity_info_data")


OnnxModelsSensitivityInfoFixture = NamedTuple(
    "OnnxModelsSensitivityInfoFixture",
    [
        ("model_path", str),
        ("loss_approx_path", str),
        ("loss_one_shot_path", str),
        ("perf_path", str),
        ("model_name", str),
        ("sensitivity_info_approx_path", str),
        ("sensitivity_info_one_shot_path", str),
    ],
)


@pytest.fixture(scope="session")
def onnx_models_sensitivity_info(
    onnx_models_with_analysis: OnnxModelAnalysisFixture,
) -> OnnxModelsSensitivityInfoFixture:
    model_path = onnx_models_with_analysis.model_path
    model_name = onnx_models_with_analysis.model_name
    loss_approx_path = onnx_models_with_analysis.loss_approx_path
    loss_one_shot_path = onnx_models_with_analysis.loss_one_shot_path
    perf_path = onnx_models_with_analysis.perf_path

    sensitivity_info_approx_path = os.path.join(
        SENSITIVITY_INFO_DIR, "{}_config_from_loss_approx".format(model_name)
    )
    sensitivity_info_one_shot_path = os.path.join(
        SENSITIVITY_INFO_DIR, "{}_config_from_loss_one_shot".format(model_name)
    )

    if GENERATE_TEST_FILES:
        sensitivity_info_approx = SensitivityModelInfo.from_sensitivities(
            model_path, perf_path, loss_approx_path
        )

        sensitivity_info_approx.save_json(sensitivity_info_approx_path)
        sensitivity_info_approx.save_csv(sensitivity_info_approx_path)

        sensitivity_info_one_shot = SensitivityModelInfo.from_sensitivities(
            model_path, perf_path, loss_one_shot_path
        )

        sensitivity_info_one_shot.save_json(sensitivity_info_one_shot_path)
        sensitivity_info_one_shot.save_csv(sensitivity_info_one_shot_path)
    return OnnxModelsSensitivityInfoFixture(
        model_path,
        loss_approx_path,
        loss_one_shot_path,
        perf_path,
        model_name,
        sensitivity_info_approx_path,
        sensitivity_info_one_shot_path,
    )


NodeInfoFixture = NamedTuple(
    "NodeInfoFixture",
    [
        ("node", NodeAnalyzer),
        ("loss_result", Dict[str, PruningSensitivityResult]),
        ("perf_result", Dict[str, PruningSensitivityResult]),
        ("loss_bucket", Union[Tuple[int, float], None]),
        ("perf_bucket", Union[Tuple[int, float], None]),
    ],
)


@pytest.fixture(
    params=[
        (
            {
                "id": "7",
                "op_type": "Conv",
                "input_names": ["input"],
                "output_names": ["7"],
                "input_shapes": [[16, 3, 3, 3]],
                "output_shapes": [[16, 16, 2, 2]],
                "params": 448,
                "prunable": True,
                "prunable_params": 432,
                "prunable_params_zeroed": 0,
                "flops": None,
                "weight_name": "seq.conv1.weight",
                "weight_shape": [16, 3, 3, 3],
                "bias_name": "seq.conv1.bias",
                "bias_shape": [16],
                "attributes": {
                    "dilations": [1, 1],
                    "group": 1,
                    "kernel_shape": [3, 3],
                    "pads": [1, 1, 1, 1],
                    "strides": [2, 2],
                },
            },
            {
                "id": "7",
                "name": "7",
                "index": 0,
                "baseline_measurement_index": 0,
                "baseline_measurement_key": None,
                "sparse_measurements": {0.0: [0.0], 0.8: [0.5]},
            },
            {
                "id": "7",
                "name": "7",
                "index": 0,
                "baseline_measurement_index": 0,
                "baseline_measurement_key": None,
                "sparse_measurements": {0.0: [0.0], 0.8: [0.05]},
            },
            (1, 0.5),
            (-1, 0.05),
        )
    ]
)
def node_info(request) -> NodeInfoFixture:
    node_dict, loss_result, perf_result, loss_bucket, perf_bucket = request.param
    node = NodeAnalyzer(None, None, **node_dict)
    loss_result = PruningSensitivityResult.from_dict(loss_result)
    perf_result = PruningSensitivityResult.from_dict(perf_result)
    return NodeInfoFixture(node, loss_result, perf_result, loss_bucket, perf_bucket)


def test_sensitivity_node_info(node_info: NodeInfoFixture):
    sensitivity_node_info = SensitivityNodeInfo(node_info.node)
    assert sensitivity_node_info.analyzer == node_info.node

    loss_id = "loss_id"
    sensitivity_node_info.add_loss_result(node_info.loss_result, loss_id)
    sensitivity_node_info.set_loss_bucket(node_info.loss_bucket)
    assert sensitivity_node_info.loss_results[loss_id] == node_info.loss_result
    assert sensitivity_node_info.loss_bucket == node_info.loss_bucket

    perf_id = "perf_id"
    sensitivity_node_info.add_perf_result(node_info.perf_result, perf_id)
    sensitivity_node_info.set_perf_bucket(node_info.perf_bucket)
    assert sensitivity_node_info.perf_results[perf_id] == node_info.perf_result
    assert sensitivity_node_info.perf_bucket == node_info.perf_bucket

    assert sensitivity_node_info.dict() == {
        "analyzer": node_info.node.dict(),
        "perf_results": [(perf_id, node_info.perf_result.dict())],
        "loss_results": [(loss_id, node_info.loss_result.dict())],
        "perf_bucket": node_info.perf_bucket,
        "loss_bucket": node_info.loss_bucket,
    }

    sensitivity_node_info.remove_loss_result(loss_id)
    assert loss_id not in sensitivity_node_info.loss_results

    sensitivity_node_info.remove_perf_result(perf_id)
    assert perf_id not in sensitivity_node_info.perf_results


def _load_sensitivity_info_files(
    sensitivity_info_path: str,
) -> Tuple[Union[Dict, List]]:
    json_path = os.path.join(SENSITIVITY_INFO_DIR, sensitivity_info_path) + ".json"
    with open(json_path) as json_file:
        sensitivity_info_json = json.load(json_file)

    csv_path = os.path.join(SENSITIVITY_INFO_DIR, sensitivity_info_path) + ".csv"
    sensitivity_info_csv = []
    with open(csv_path) as csv_file:
        for row in csv.reader(csv_file):
            sensitivity_info_csv.append(row)
    return sensitivity_info_json, sensitivity_info_csv


def _load_sensitivity_info_fields_from_json(
    sensitivity_info_json: Dict[str, Union[List, Dict]]
):
    analyzer = ModelAnalyzer.from_dict(sensitivity_info_json["analyzer"])
    perf_analysis = OrderedDict(
        [
            (key, PruningPerfSensitivityAnalysis.from_dict(analysis))
            for key, analysis in sensitivity_info_json["perf_analysis"]
        ]
    )

    loss_analysis = OrderedDict(
        [
            (key, PruningLossSensitivityAnalysis.from_dict(analysis))
            for key, analysis in sensitivity_info_json["loss_analysis"]
        ]
    )

    perf_buckets = OrderedDict(sensitivity_info_json["perf_buckets"])
    loss_buckets = OrderedDict(sensitivity_info_json["loss_buckets"])
    return analyzer, perf_analysis, loss_analysis, perf_buckets, loss_buckets


def _test_sensitivity_info_properties(
    sensitivity_info, analyzer, perf_analysis, loss_analysis, perf_buckets, loss_buckets
):
    # Test properties
    assert sensitivity_info.analyzer == analyzer
    actual_perf_analysis = sensitivity_info.perf_analysis
    for actual_key, expected_key in zip(actual_perf_analysis, perf_analysis):
        assert (
            actual_perf_analysis[actual_key].dict()
            == perf_analysis[expected_key].dict()
        )

    actual_loss_analysis = sensitivity_info.loss_analysis
    for actual_key, expected_key in zip(actual_loss_analysis, loss_analysis):
        assert (
            actual_loss_analysis[actual_key].dict()
            == loss_analysis[expected_key].dict()
        )

    for key in perf_buckets:
        assert list(sensitivity_info.perf_buckets[key]) == list(perf_buckets[key])
    for key in loss_buckets:
        assert list(sensitivity_info.loss_buckets[key]) == list(loss_buckets[key])


def _test_sensitivity_info_csv(sensitivity_info, sensitivity_info_csv):
    # Test csv
    for index, csv_row in enumerate(sensitivity_info_csv):
        if index == 0:
            assert csv_row == sensitivity_info.table_headers()
        elif index == 1:
            assert csv_row == sensitivity_info.table_titles()
        else:
            table_node_row = sensitivity_info.table_node_row(csv_row[1])
            for cvs_elem, table_elem in zip(csv_row, table_node_row):
                assert (
                    cvs_elem == str(table_elem) or cvs_elem == "" and table_elem is None
                )


def _test_sensitivity_info_saving(
    sensitivity_info, sensitivity_info_json, sensitivity_info_csv
):
    # Test saving
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_json_path = os.path.join(temp_dir, "temp.json")
        sensitivity_info.save_json(temp_json_path)
        with open(temp_json_path) as temp_file:
            generated_json = json.load(temp_file)
            assert generated_json["analyzer"] == sensitivity_info_json["analyzer"]
            assert (
                generated_json["perf_buckets"] == sensitivity_info_json["perf_buckets"]
            )
            assert (
                generated_json["loss_buckets"] == sensitivity_info_json["loss_buckets"]
            )

            # Because of IDs being generated, need to check that the contents match minus ID

            for generated_perf_analysis, actual_perf_analysis in zip(
                generated_json["perf_analysis"], sensitivity_info_json["perf_analysis"]
            ):
                assert generated_perf_analysis[1] == actual_perf_analysis[1]

            for generated_loss_analysis, actual_loss_analysis in zip(
                generated_json["loss_analysis"], sensitivity_info_json["loss_analysis"]
            ):
                assert generated_loss_analysis[1] == actual_loss_analysis[1]

            for (
                (generated_node_id, generated_node),
                (actual_node_id, actual_node),
            ) in zip(generated_json["nodes"], sensitivity_info_json["nodes"]):
                assert generated_node_id == actual_node_id
                assert generated_node["analyzer"] == actual_node["analyzer"]
                assert generated_node["perf_bucket"] == actual_node["perf_bucket"]
                assert generated_node["loss_bucket"] == actual_node["loss_bucket"]

                for generated_node_perf_result, actual_node_perf_result in zip(
                    generated_node["perf_results"], actual_node["perf_results"]
                ):
                    assert generated_node_perf_result[1] == actual_node_perf_result[1]

                for generated_node_loss_result, actual_node_loss_result in zip(
                    generated_node["loss_results"], actual_node["loss_results"]
                ):
                    assert generated_node_loss_result[1] == actual_node_loss_result[1]

        temp_csv_path = os.path.join(temp_dir, "temp.csv")
        sensitivity_info.save_csv(temp_csv_path)
        with open(temp_csv_path) as temp_file:
            for row, expected_row in zip(csv.reader(temp_file), sensitivity_info_csv):
                assert row == expected_row


def _test_sensitivity_info_mutators(
    sensitivity_info, perf_analysis, loss_analysis, perf_buckets, loss_buckets
):
    # Test mutating analysis
    first_id = list(perf_analysis.keys())[0]
    analysis_id = sensitivity_info.add_perf_analysis(perf_analysis[first_id])
    assert analysis_id in sensitivity_info.perf_analysis
    for _, node in sensitivity_info.nodes.items():
        assert analysis_id in node.perf_results or len(node.perf_results) == 0

    first_id = list(loss_analysis.keys())[0]
    analysis_id = sensitivity_info.add_loss_analysis(loss_analysis[first_id])
    assert analysis_id in sensitivity_info.loss_analysis
    for _, node in sensitivity_info.nodes.items():
        assert analysis_id in node.loss_results or len(node.loss_results) == 0

    # Test mutating buckets
    zero_buckets = OrderedDict([(key, (0, 0)) for key in perf_buckets])

    sensitivity_info.set_perf_buckets(zero_buckets)
    assert sensitivity_info.perf_buckets == zero_buckets
    for key, node in sensitivity_info.nodes.items():
        assert (
            node.perf_bucket == (0, 0)
            if key in sensitivity_info.perf_buckets
            else node.perf_bucket is None
        )

    sensitivity_info.set_loss_buckets(zero_buckets)
    assert sensitivity_info.loss_buckets == zero_buckets
    for key, node in sensitivity_info.nodes.items():
        assert (
            node.loss_bucket == (0, 0)
            if key in sensitivity_info.loss_buckets
            else node.loss_bucket is None
        )


def _test_sensitivity_info(
    sensitivity_info, sensitivity_info_json, sensitivity_info_csv
):
    (
        analyzer,
        perf_analysis,
        loss_analysis,
        perf_buckets,
        loss_buckets,
    ) = _load_sensitivity_info_fields_from_json(sensitivity_info_json)

    _test_sensitivity_info_properties(
        sensitivity_info,
        analyzer,
        perf_analysis,
        loss_analysis,
        perf_buckets,
        loss_buckets,
    )

    _test_sensitivity_info_csv(sensitivity_info, sensitivity_info_csv)

    _test_sensitivity_info_saving(
        sensitivity_info, sensitivity_info_json, sensitivity_info_csv
    )

    _test_sensitivity_info_mutators(
        sensitivity_info, perf_analysis, loss_analysis, perf_buckets, loss_buckets
    )


def test_sensitivity_info_data_from_file(
    onnx_models_sensitivity_info: OnnxModelsSensitivityInfoFixture,
):
    sensitivity_info_paths = [
        onnx_models_sensitivity_info.sensitivity_info_approx_path,
        onnx_models_sensitivity_info.sensitivity_info_one_shot_path,
    ]

    for sensitivity_info_path in sensitivity_info_paths:
        sensitivity_info_json, sensitivity_info_csv = _load_sensitivity_info_files(
            sensitivity_info_path
        )

        sensitivity_info_from_dict = SensitivityModelInfo.from_dict(
            sensitivity_info_json
        )
        _test_sensitivity_info(
            sensitivity_info_from_dict, sensitivity_info_json, sensitivity_info_csv
        )

        sensitivity_info_from_json = SensitivityModelInfo.load_json(
            os.path.join(SENSITIVITY_INFO_DIR, sensitivity_info_path) + ".json"
        )
        _test_sensitivity_info(
            sensitivity_info_from_json, sensitivity_info_json, sensitivity_info_csv
        )


def test_sensitivity_info_data_from_sensitivities(
    onnx_models_sensitivity_info: OnnxModelsSensitivityInfoFixture,
):
    sensitivity_info_path_pairs = [
        (
            onnx_models_sensitivity_info.loss_approx_path,
            onnx_models_sensitivity_info.sensitivity_info_approx_path,
        ),
        (
            onnx_models_sensitivity_info.loss_one_shot_path,
            onnx_models_sensitivity_info.sensitivity_info_one_shot_path,
        ),
    ]
    model_path = onnx_models_sensitivity_info.model_path
    perf_path = onnx_models_sensitivity_info.perf_path

    for loss_path, sensitivity_info_path in sensitivity_info_path_pairs:
        sensitivity_info = SensitivityModelInfo.from_sensitivities(
            model_path, perf_path, loss_path
        )
        sensitivity_info_json, sensitivity_info_csv = _load_sensitivity_info_files(
            sensitivity_info_path
        )

        _test_sensitivity_info(
            sensitivity_info, sensitivity_info_json, sensitivity_info_csv
        )
