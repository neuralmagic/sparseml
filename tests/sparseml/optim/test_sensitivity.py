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

import os
import tempfile

from sparseml.optim import PruningLossSensitivityAnalysis


def test_ks_loss_sensitivity_analysis():
    test_samples = [
        {
            "id": "165",
            "name": "input.conv.weight",
            "index": 0,
            "sparse_measurements": [
                [0.0, [0.0, 0.0]],
                [0.2, [0.0, 0.0]],
                [0.4, [0.0, 0.0]],
                [0.6, [1.4423741959035397e-05, 1.4478888260782696e-05]],
                [0.7, [0.0003933242114726454, 0.0004009161493740976]],
                [0.8, [0.002293953439220786, 0.002319519640877843]],
                [0.85, [0.0038978520315140486, 0.003912879154086113]],
                [0.9, [0.0024482859298586845, 0.0024934178218245506]],
                [0.95, [0.0034274826757609844, 0.003474951023235917]],
                [0.99, [0.01961200125515461, 0.01976676657795906]],
            ],
            "averages": {
                "0.0": 0.0,
                "0.2": 0.0,
                "0.4": 0.0,
                "0.6": 1.4451315109909046e-05,
                "0.7": 0.0003971201804233715,
                "0.8": 0.0023067365400493145,
                "0.85": 0.0039053655928000808,
                "0.9": 0.0024708518758416176,
                "0.95": 0.0034512168494984508,
                "0.99": 0.019689383916556835,
            },
            "sparse_average": 0.321111756313514,
            "sparse_integral": 0.0010827882658031743,
            "sparse_comparison": 0.0024708518758416176,
        },
        {
            "id": "168",
            "name": "sections.0.0.depth.conv.weight",
            "index": 1,
            "sparse_measurements": [
                [0.0, [0.0, 0.0]],
                [0.2, [0.0, 0.0]],
                [0.4, [0.0, 0.0]],
                [0.6, [0.001039206050336361, 0.0010454836301505566]],
                [0.7, [0.0013909710105508566, 0.001424945192411542]],
                [0.8, [0.005448495503515005, 0.005359217524528503]],
                [0.85, [0.0024713557213544846, 0.0024402134586125612]],
                [0.9, [0.002173610497266054, 0.002142299897968769]],
                [0.95, [0.00811902154237032, 0.008098462596535683]],
                [0.99, [0.06605849775951356, 0.06613280531018972]],
            ],
            "averages": {
                "0.0": 0.0,
                "0.2": 0.0,
                "0.4": 0.0,
                "0.6": 0.0010423448402434587,
                "0.7": 0.0014079581014811993,
                "0.8": 0.005403856514021754,
                "0.85": 0.002455784589983523,
                "0.9": 0.0021579551976174116,
                "0.95": 0.008108742069453001,
                "0.99": 0.06609565153485164,
            },
            "sparse_average": 0.32383361464238264,
            "sparse_integral": 0.002619930187938736,
            "sparse_comparison": 0.0021579551976174116,
        },
        {
            "id": "171",
            "name": "sections.0.0.point.conv.weight",
            "index": 2,
            "sparse_measurements": [
                [0.0, [0.0, 0.0]],
                [0.2, [0.0, 0.0]],
                [0.4, [0.0, 0.0]],
                [0.6, [-9.29515908687506e-10, -1.6410388603560477e-09]],
                [0.7, [5.841321808475186e-07, 6.879848797325394e-07]],
                [0.8, [0.00011716883454937488, 0.00011542218999238685]],
                [0.85, [3.637020199676044e-05, 3.672009552246891e-05]],
                [0.9, [0.00020571750064846128, 0.0002049835748039186]],
                [0.95, [0.0002617501886561513, 0.00026932702166959643]],
                [0.99, [0.0006772654596716166, 0.0006722339312545955]],
            ],
            "averages": {
                "0.0": 0.0,
                "0.2": 0.0,
                "0.4": 0.0,
                "0.6": -1.2852773845217769e-09,
                "0.7": 6.36058530290029e-07,
                "0.8": 0.00011629551227088086,
                "0.85": 3.6545148759614676e-05,
                "0.9": 0.00020535053772618994,
                "0.95": 0.00026553860516287386,
                "0.99": 0.000674749695463106,
            },
            "sparse_average": 0.3195649557136318,
            "sparse_integral": 4.6324591947619074e-05,
            "sparse_comparison": 0.00020535053772618994,
        },
    ]

    analysis = PruningLossSensitivityAnalysis()

    for test in test_samples:
        for sparse_measure in test["sparse_measurements"]:
            for meas in sparse_measure[1]:
                analysis.add_result(
                    test["id"],
                    test["name"],
                    test["index"],
                    sparse_measure[0],
                    meas,
                    baseline=False,
                )

        comp = analysis.results[-1]

        assert test["id"] == comp.id_
        assert test["name"] == comp.name
        assert test["index"] == comp.index
        assert len(test["sparse_measurements"]) == len(comp.sparse_measurements)
        assert len(test["averages"]) == len(comp.averages)
        assert abs(test["sparse_average"] - comp.sparse_average) < 1e-5
        assert abs(test["sparse_integral"] - comp.sparse_integral) < 1e-5
        assert abs(test["sparse_comparison"] - comp.sparse_comparison()) < 1e-5

    path = os.path.join(tempfile.gettempdir(), "ks-sens-analysis.json")
    analysis.save_json(path)

    json_analysis = analysis.load_json(path)

    for index, test in enumerate(test_samples):
        comp = json_analysis.results[index]

        assert test["id"] == comp.id_
        assert test["name"] == comp.name
        assert test["index"] == comp.index
        assert len(test["sparse_measurements"]) == len(comp.sparse_measurements)
        assert len(test["averages"]) == len(comp.averages)
        assert abs(test["sparse_average"] - comp.sparse_average) < 1e-5
        assert abs(test["sparse_integral"] - comp.sparse_integral) < 1e-5
        assert abs(test["sparse_comparison"] - comp.sparse_comparison()) < 1e-5

    path = os.path.join(
        tempfile.gettempdir(), "ks-sens-analysis-integral-normalized.png"
    )
    analysis.plot(path, plot_integral=True)
    assert os.path.exists(path)

    path = os.path.join(
        tempfile.gettempdir(), "ks-sens-analysis-integral-normalized.png"
    )
    analysis.plot(path, plot_integral=True)
    assert os.path.exists(path)

    path = os.path.join(tempfile.gettempdir(), "ks-sens-analysis-integral.png")
    analysis.plot(path, plot_integral=True, normalize=False)
    assert os.path.exists(path)

    path = os.path.join(tempfile.gettempdir(), "ks-sens-analysis-avg-normalized.png")
    analysis.plot(path, plot_integral=False)
    assert os.path.exists(path)

    path = os.path.join(tempfile.gettempdir(), "ks-sens-analysis-avg.png")
    analysis.plot(path, plot_integral=False, normalize=False)
    assert os.path.exists(path)
