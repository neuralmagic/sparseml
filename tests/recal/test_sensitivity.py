import tempfile
import os

from neuralmagicML.recal import KSLossSensitivityAnalysis


def test_ks_loss_sensitivity_analysis_load():
    test_samples = [
        {
            "param": "input.conv.weight",
            "index": 11,
            "sparse_measurements": [
                [
                    0.0,
                    [
                        1.065279483795166,
                        1.065279483795166,
                        0.3132499158382416,
                        0.3132499158382416,
                    ],
                ],
                [
                    0.05,
                    [
                        1.065279483795166,
                        1.065279483795166,
                        0.3132499158382416,
                        0.3132499158382416,
                    ],
                ],
                [
                    0.2,
                    [
                        1.065279483795166,
                        1.065279483795166,
                        0.3132499158382416,
                        0.3132499158382416,
                    ],
                ],
                [
                    0.4,
                    [
                        1.065279483795166,
                        1.065279483795166,
                        0.3132499158382416,
                        0.3132499158382416,
                    ],
                ],
                [
                    0.6,
                    [
                        1.0599160194396973,
                        1.0599160194396973,
                        0.29898542165756226,
                        0.29898542165756226,
                    ],
                ],
                [
                    0.7,
                    [
                        1.0312217473983765,
                        1.0312217473983765,
                        0.3881630599498749,
                        0.3881630599498749,
                    ],
                ],
                [
                    0.8,
                    [
                        0.9448628425598145,
                        0.9448628425598145,
                        0.5429320335388184,
                        0.5429320335388184,
                    ],
                ],
                [
                    0.9,
                    [
                        1.3083751201629639,
                        1.3083751201629639,
                        3.844062566757202,
                        3.844062566757202,
                    ],
                ],
                [
                    0.95,
                    [
                        5.4708051681518555,
                        5.4708051681518555,
                        6.9318132400512695,
                        6.9318132400512695,
                    ],
                ],
                [
                    0.975,
                    [
                        6.424170970916748,
                        6.424170970916748,
                        6.58629035949707,
                        6.58629035949707,
                    ],
                ],
                [
                    0.99,
                    [
                        6.712153434753418,
                        6.712153434753418,
                        7.9098219871521,
                        7.9098219871521,
                    ],
                ],
            ],
            "sparse_averages": [
                [0.0, 0.6892646998167038],
                [0.05, 0.6892646998167038],
                [0.2, 0.6892646998167038],
                [0.4, 0.6892646998167038],
                [0.6, 0.6794507205486298],
                [0.7, 0.7096924036741257],
                [0.8, 0.7438974380493164],
                [0.9, 2.576218843460083],
                [0.95, 6.2013092041015625],
                [0.975, 6.505230665206909],
                [0.99, 7.310987710952759],
            ],
            "sparse_loss_avg": 2.4985314350236547,
            "sparse_loss_integral": 1.202611471712589,
        },
        {
            "param": "sections.0.0.depth.conv.weight",
            "index": 11,
            "sparse_measurements": [
                [
                    0.0,
                    [
                        1.065279483795166,
                        1.065279483795166,
                        0.3132499158382416,
                        0.3132499158382416,
                    ],
                ],
                [
                    0.05,
                    [
                        1.065279483795166,
                        1.065279483795166,
                        0.3132499158382416,
                        0.3132499158382416,
                    ],
                ],
                [
                    0.2,
                    [
                        1.065279483795166,
                        1.065279483795166,
                        0.3132499158382416,
                        0.3132499158382416,
                    ],
                ],
                [
                    0.4,
                    [
                        1.065279483795166,
                        1.065279483795166,
                        0.3132499158382416,
                        0.3132499158382416,
                    ],
                ],
                [
                    0.6,
                    [
                        0.9483986496925354,
                        0.9483986496925354,
                        0.3595724403858185,
                        0.3595724403858185,
                    ],
                ],
                [
                    0.7,
                    [
                        1.4048981666564941,
                        1.4048981666564941,
                        3.250788688659668,
                        3.250788688659668,
                    ],
                ],
                [
                    0.8,
                    [
                        1.3648375272750854,
                        1.3648375272750854,
                        4.859118461608887,
                        4.859118461608887,
                    ],
                ],
                [
                    0.9,
                    [
                        2.538297653198242,
                        2.538297653198242,
                        6.704424858093262,
                        6.704424858093262,
                    ],
                ],
                [
                    0.95,
                    [
                        3.6454005241394043,
                        3.6454005241394043,
                        7.289975166320801,
                        7.289975166320801,
                    ],
                ],
                [
                    0.975,
                    [
                        6.85790491104126,
                        6.85790491104126,
                        11.210531234741211,
                        11.210531234741211,
                    ],
                ],
                [
                    0.99,
                    [
                        9.771215438842773,
                        9.771215438842773,
                        11.361690521240234,
                        11.361690521240234,
                    ],
                ],
            ],
            "sparse_averages": [
                [0.0, 0.6892646998167038],
                [0.05, 0.6892646998167038],
                [0.2, 0.6892646998167038],
                [0.4, 0.6892646998167038],
                [0.6, 0.6539855450391769],
                [0.7, 2.327843427658081],
                [0.8, 3.111977994441986],
                [0.9, 4.621361255645752],
                [0.95, 5.4676878452301025],
                [0.975, 9.034218072891235],
                [0.99, 10.566452980041504],
            ],
            "sparse_loss_avg": 3.503689629110423,
            "sparse_loss_integral": 1.7982854710519314,
        },
    ]

    analysis = KSLossSensitivityAnalysis()

    for test in test_samples:
        analysis.add_result(test["param"], test["index"], test["sparse_measurements"])
        comp = analysis.results[-1]

        assert test["param"] == comp["param"]
        assert test["index"] == comp["index"]
        assert len(test["sparse_measurements"]) == len(comp["sparse_measurements"])
        assert len(test["sparse_averages"]) == len(comp["sparse_averages"])
        assert abs(test["sparse_loss_avg"] - comp["sparse_loss_avg"]) < 1e-5
        assert abs(test["sparse_loss_integral"] - comp["sparse_loss_integral"]) < 1e-5

    path = os.path.join(tempfile.gettempdir(), "ks-sens-analysis.json")
    analysis.save_json(path)

    json_analysis = analysis.load_json(path)

    for index, test in enumerate(test_samples):
        comp = json_analysis.results[index]

        assert test["param"] == comp["param"]
        assert test["index"] == comp["index"]
        assert len(test["sparse_measurements"]) == len(comp["sparse_measurements"])
        assert len(test["sparse_averages"]) == len(comp["sparse_averages"])
        assert abs(test["sparse_loss_avg"] - comp["sparse_loss_avg"]) < 1e-5
        assert abs(test["sparse_loss_integral"] - comp["sparse_loss_integral"]) < 1e-5

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
