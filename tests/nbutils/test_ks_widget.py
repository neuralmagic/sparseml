from neuralmagicML.nbutils import (
    KSWidgetContainer,
    PruningEpochWidget,
    PruneLayerWidget,
    PruningLayersWidget,
)


def test_pruning_epoch_widget():
    widget = PruningEpochWidget(
        start_epoch=0, end_epoch=10, total_epochs=50, max_epochs=100
    )
    widget.create()


def test_prune_layer_widget():
    widget = PruneLayerWidget(
        name="name",
        desc="desc",
        enabled=True,
        end_sparsity=0.8,
        loss_sens_analysis=None,
    )
    widget.create()


def test_pruning_layers_widiget():
    widget = PruningLayersWidget(
        layer_names=["layer_one", "layer_two"],
        layer_descs=["desc_one", "desc_two"],
        layer_enables=[True, False],
        layer_sparsities=[0.8, 0.9],
        loss_sens_analysis=None,
    )
    widget.create()


def test_ks_widget_container():
    epoch_widget = PruningEpochWidget(
        start_epoch=0, end_epoch=10, total_epochs=50, max_epochs=100
    )
    layers_widget = PruningLayersWidget(
        layer_names=["layer_one", "layer_two"],
        layer_descs=["desc_one", "desc_two"],
        layer_enables=[True, False],
        layer_sparsities=[0.8, 0.9],
        loss_sens_analysis=None,
    )
    container = KSWidgetContainer(epoch_widget, layers_widget)
    container.create()
