from sparseml.utilsnb import (
    KSWidgetContainer,
    PruningEpochWidget,
    PruneParamWidget,
    PruningParamsWidget,
)


def test_pruning_epoch_widget():
    widget = PruningEpochWidget(
        start_epoch=0, end_epoch=10, total_epochs=50, max_epochs=100
    )
    widget.create()


def test_prune_param_widget():
    widget = PruneParamWidget(
        name="name",
        desc="desc",
        enabled=True,
        end_sparsity=0.8,
        loss_sens_analysis=None,
    )
    widget.create()


def test_pruning_params_widiget():
    widget = PruningParamsWidget(
        param_names=["param_one", "param_two"],
        param_descs=["desc_one", "desc_two"],
        param_enables=[True, False],
        param_sparsities=[0.8, 0.9],
        loss_sens_analysis=None,
    )
    widget.create()


def test_ks_widget_container():
    epoch_widget = PruningEpochWidget(
        start_epoch=0, end_epoch=10, total_epochs=50, max_epochs=100
    )
    params_widget = PruningParamsWidget(
        param_names=["param_one", "param_two"],
        param_descs=["desc_one", "desc_two"],
        param_enables=[True, False],
        param_sparsities=[0.8, 0.9],
        loss_sens_analysis=None,
    )
    container = KSWidgetContainer(epoch_widget, params_widget)
    container.create()
