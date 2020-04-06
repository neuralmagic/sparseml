import os
import tempfile

from neuralmagicML.recal import AnalyzedLayerDesc


def test_layer_descs():
    descs = [
        AnalyzedLayerDesc("layer1", "Linear"),
        AnalyzedLayerDesc("layer2", "Conv2d"),
    ]
    save_path = os.path.join(tempfile.gettempdir(), "layer_descs.json")

    AnalyzedLayerDesc.save_descs(descs, save_path)
    loaded_descs = AnalyzedLayerDesc.load_descs(save_path)

    for desc, loaded_desc in zip(descs, loaded_descs):
        assert desc.name == loaded_desc.name
        assert desc.type_ == loaded_desc.type_
