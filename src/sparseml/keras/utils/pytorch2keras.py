import re

import numpy as np
from tensorflow import keras

from sparseml.pytorch.models import ModelRegistry as PTModelRegistry


def get_keras_model(model_name: str):
    print("Loading Keras model {}".format(model_name))
    if model_name == "resnet50":
        model = keras.applications.ResNet50(include_top=True, weights=None)
    else:
        raise ValueError("Unknown model")
    print([layer.name for layer in model.layers])
    return model


def get_pytorch_model(model_name: str, model_type: str):
    print("Loading PyTorch model {}, {}".format(model_name, model_type))
    model = PTModelRegistry.create(model_name, pretrained=model_type)
    print(model.state_dict().keys())
    return model


def convert_model(pt_model, kr_model, layer_mapping):
    pt_model_state_dict = pt_model.state_dict()
    for layer in kr_model.layers:
        pt_layer_name = get_pytorch_name(layer.name, layer_mapping)
        if pt_layer_name is None:
            print(
                "W: {} layer in Keras will not be updated from Pytorch".format(
                    layer.name
                )
            )
            continue

        if isinstance(layer, keras.layers.Conv2D):
            convert_conv_layer(layer, pt_layer_name, pt_model_state_dict)
        elif isinstance(layer, keras.layers.BatchNormalization):
            convert_bn_layer(layer, pt_layer_name, pt_model_state_dict)
        elif isinstance(layer, keras.layers.Dense):
            convert_dense_layer(layer, pt_layer_name, pt_model_state_dict)


def set_kr_layer_weights(kr_layer, pt_weights):
    kr_weights = kr_layer.weights
    assert len(kr_weights) == len(pt_weights)
    for kr_w, pt_w in zip(kr_weights, pt_weights):
        assert kr_w.shape == pt_w.shape
    kr_layer.set_weights(pt_weights)


def convert_conv_layer(kr_layer, pt_layer_name, pt_model_state_dict):
    print("Converting Conv2D layer {}...".format(kr_layer.name))
    kr_weights = kr_layer.get_weights()

    pt_weight = pt_model_state_dict["{}.weight".format(pt_layer_name)].numpy()
    pt_weight = np.transpose(pt_weight, (2, 3, 1, 0))
    pt_weight = np.ascontiguousarray(pt_weight)

    pt_bias_name = "{}.bias".format(pt_layer_name)
    pt_bias = pt_model_state_dict.get(pt_bias_name, np.zeros_like(kr_weights[1]))

    set_kr_layer_weights(kr_layer, [pt_weight, pt_bias])
    print(">> Done\n")


def convert_bn_layer(kr_layer, pt_layer_name, pt_model_state_dict):
    print("Converting Batch Norm layer {}...".format(kr_layer.name))
    kr_weights = kr_layer.get_weights()

    pt_gamma = pt_model_state_dict["{}.weight".format(pt_layer_name)]
    pt_beta = pt_model_state_dict["{}.bias".format(pt_layer_name)]
    pt_mean = pt_model_state_dict["{}.running_mean".format(pt_layer_name)]
    pt_var = pt_model_state_dict["{}.running_var".format(pt_layer_name)]
    set_kr_layer_weights(kr_layer, [pt_gamma, pt_beta, pt_mean, pt_var])
    print(">> Done\n")


def convert_dense_layer(kr_layer, pt_layer_name, pt_model_state_dict):
    print("Converting Dense layer {}...".format(kr_layer.name))
    kr_weights = kr_layer.get_weights()

    pt_weight = pt_model_state_dict["{}.weight".format(pt_layer_name)].numpy()
    pt_weight = np.transpose(pt_weight, (1, 0))
    pt_weight = np.ascontiguousarray(pt_weight)

    pt_bias_name = "{}.bias".format(pt_layer_name)
    pt_bias = pt_model_state_dict.get(pt_bias_name, np.zeros_like(kr_weights[1]))

    set_kr_layer_weights(kr_layer, [pt_weight, pt_bias])
    print(">> Done\n")


def get_pytorch_name(kr_name, layer_mapping):
    for key in layer_mapping.keys():
        match = re.match(key, kr_name)
        if not match:
            continue
        if layer_mapping[key]:
            if match.groups():
                return layer_mapping[key].format(*match.groups())
            else:
                return layer_mapping[key]
    return None


resnet50_mapping = {
    # Input
    "conv1_conv": "input.conv",
    "conv1_bn": "input.bn",
    ##########################
    # conv2: 3 blocks
    # Block 1
    "conv2_block1_0_conv": "sections.0.0.identity.conv",
    "conv2_block1_0_bn": "sections.0.0.identity.bn",
    "conv2_block1_([1-9]+)_conv": "sections.0.0.conv{}",
    "conv2_block1_([1-9]+)_bn": "sections.0.0.bn{}",
    # Block 2
    "conv2_block2_0_conv": "sections.0.1.identity.conv",
    "conv2_block2_0_bn": "sections.0.1.identity.bn",
    "conv2_block2_([1-9]+)_conv": "sections.0.1.conv{}",
    "conv2_block2_([1-9]+)_bn": "sections.0.1.bn{}",
    # Block 3
    "conv2_block3_0_conv": "sections.0.2.identity.conv",
    "conv2_block3_0_bn": "sections.0.2.identity.bn",
    "conv2_block3_([1-9]+)_conv": "sections.0.2.conv{}",
    "conv2_block3_([1-9]+)_bn": "sections.0.2.bn{}",
    ###########################
    # conv3: 4 blocks
    # Block 1
    "conv3_block1_0_conv": "sections.1.0.identity.conv",
    "conv3_block1_0_bn": "sections.1.0.identity.bn",
    "conv3_block1_([1-9]+)_conv": "sections.1.0.conv{}",
    "conv3_block1_([1-9]+)_bn": "sections.1.0.bn{}",
    # Block 2
    "conv3_block2_0_conv": "sections.1.1.identity.conv",
    "conv3_block2_0_bn": "sections.1.1.identity.bn",
    "conv3_block2_([1-9]+)_conv": "sections.1.1.conv{}",
    "conv3_block2_([1-9]+)_bn": "sections.1.1.bn{}",
    # Block 3
    "conv3_block3_0_conv": "sections.1.2.identity.conv",
    "conv3_block3_0_bn": "sections.1.2.identity.bn",
    "conv3_block3_([1-9]+)_conv": "sections.1.2.conv{}",
    "conv3_block3_([1-9]+)_bn": "sections.1.2.bn{}",
    # Block 4
    "conv3_block4_0_conv": "sections.1.3.identity.conv",
    "conv3_block4_0_bn": "sections.1.3.identity.bn",
    "conv3_block4_([1-9]+)_conv": "sections.1.3.conv{}",
    "conv3_block4_([1-9]+)_bn": "sections.1.3.bn{}",
    ############################
    # conv4: 6 blocks
    # Block 1
    "conv4_block1_0_conv": "sections.2.0.identity.conv",
    "conv4_block1_0_bn": "sections.2.0.identity.bn",
    "conv4_block1_([1-9]+)_conv": "sections.2.0.conv{}",
    "conv4_block1_([1-9]+)_bn": "sections.2.0.bn{}",
    # Block 2
    "conv4_block2_0_conv": "sections.2.1.identity.conv",
    "conv4_block2_0_bn": "sections.2.1.identity.bn",
    "conv4_block2_([1-9]+)_conv": "sections.2.1.conv{}",
    "conv4_block2_([1-9]+)_bn": "sections.2.1.bn{}",
    # Block 3
    "conv4_block3_0_conv": "sections.2.2.identity.conv",
    "conv4_block3_0_bn": "sections.2.2.identity.bn",
    "conv4_block3_([1-9]+)_conv": "sections.2.2.conv{}",
    "conv4_block3_([1-9]+)_bn": "sections.2.2.bn{}",
    # Block 4
    "conv4_block4_0_conv": "sections.2.3.identity.conv",
    "conv4_block4_0_bn": "sections.2.3.identity.bn",
    "conv4_block4_([1-9]+)_conv": "sections.2.3.conv{}",
    "conv4_block4_([1-9]+)_bn": "sections.2.3.bn{}",
    # Block 5
    "conv4_block5_0_conv": "sections.2.4.identity.conv",
    "conv4_block5_0_bn": "sections.2.4.identity.bn",
    "conv4_block5_([1-9]+)_conv": "sections.2.4.conv{}",
    "conv4_block5_([1-9]+)_bn": "sections.2.4.bn{}",
    # Block 6
    "conv4_block5_0_conv": "sections.2.5.identity.conv",
    "conv4_block5_0_bn": "sections.2.5.identity.bn",
    "conv4_block5_([1-9]+)_conv": "sections.2.5.conv{}",
    "conv4_block5_([1-9]+)_bn": "sections.2.5.bn{}",
    ############################
    # conv5: 3 blocks
    # Block 1
    "conv5_block1_0_conv": "sections.3.0.identity.conv",
    "conv5_block1_0_bn": "sections.3.0.identity.bn",
    "conv5_block1_([1-9]+)_conv": "sections.3.0.conv{}",
    "conv5_block1_([1-9]+)_bn": "sections.3.0.bn{}",
    # Block 2
    "conv5_block2_0_conv": "sections.3.1.identity.conv",
    "conv5_block2_0_bn": "sections.3.1.identity.bn",
    "conv5_block2_([1-9]+)_conv": "sections.3.1.conv{}",
    "conv5_block2_([1-9]+)_bn": "sections.3.1.bn{}",
    # Block 3
    "conv5_block3_0_conv": "sections.3.2.identity.conv",
    "conv5_block3_0_bn": "sections.3.2.identity.bn",
    "conv5_block3_([1-9]+)_conv": "sections.3.2.conv{}",
    "conv5_block3_([1-9]+)_bn": "sections.3.2.bn{}",
    # Classifier
    "predictions": "classifier.fc",
}


def main():
    pt_model = get_pytorch_model("resnet50", "base")
    kr_model = get_keras_model("resnet50")
    convert_model(pt_model, kr_model, resnet50_mapping)


if __name__ == "__main__":
    main()
