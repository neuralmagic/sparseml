import re
import os
from typing import Dict

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tensorflow import keras

from sparseml.keras.datasets import ImageNetDataset
from sparseml.keras.models import ModelRegistry as KRModelRegistry
from sparseml.keras.utils import ModelExporter
from sparseml.pytorch.models import ModelRegistry as PTModelRegistry
from sparseml.utils.datasets import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS

OUTPUT_DIR = "/hdd/src/sparseml/pytorch2keras_models/"

IMAGENET_DIR = "/hdd/datasets/ILSVRC"

IMAGE_FILE_PATH = "/hdd/datasets/ILSVRC/val/n02481823/ILSVRC2012_val_00024600.JPEG"

BN_MOMENTUM = 0.9
BN_EPSILON = 1e-5


def load_the_image():
    image = Image.open(IMAGE_FILE_PATH)
    image.load()

    init_trans = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    trans = [
        *init_trans,
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_RGB_MEANS, std=IMAGENET_RGB_STDS),
    ]
    for tran in trans:
        image = tran(image)
    image_batch = image.unsqueeze(0)
    return image_batch.numpy()


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

    assert kr_weights[0].shape == pt_weight.shape
    if len(kr_weights) > 1:
        pt_bias_name = "{}.bias".format(pt_layer_name)
        pt_bias = pt_model_state_dict.get(pt_bias_name, np.zeros_like(kr_weights[1]))
        kr_new_weights = [pt_weight, pt_bias]
    else:
        kr_new_weights = [pt_weight]

    set_kr_layer_weights(kr_layer, kr_new_weights)
    print(">> Done\n")


def convert_bn_layer(kr_layer, pt_layer_name, pt_model_state_dict):
    print("Converting Batch Norm layer {}...".format(kr_layer.name))
    kr_weights = kr_layer.get_weights()

    pt_gamma = pt_model_state_dict["{}.weight".format(pt_layer_name)]
    pt_beta = pt_model_state_dict["{}.bias".format(pt_layer_name)]
    pt_mean = pt_model_state_dict["{}.running_mean".format(pt_layer_name)]
    pt_var = pt_model_state_dict["{}.running_var".format(pt_layer_name)]
    set_kr_layer_weights(kr_layer, [pt_gamma, pt_beta, pt_mean, pt_var])

    kr_layer.epsilon = BN_EPSILON
    kr_layer.momentum = BN_MOMENTUM

    print(">> Done\n")


def convert_dense_layer(kr_layer, pt_layer_name, pt_model_state_dict):
    print("Converting Dense layer {}...".format(kr_layer.name))
    kr_weights = kr_layer.get_weights()

    pt_weight = pt_model_state_dict["{}.weight".format(pt_layer_name)].numpy()
    pt_weight = np.transpose(pt_weight, (1, 0))
    pt_weight = np.ascontiguousarray(pt_weight)
    assert kr_weights[0].shape == pt_weight.shape

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


def verify_keras_model(kr_model):
    batch_size = 128
    val_dataset = ImageNetDataset(IMAGENET_DIR, train=False)
    val_dataset = val_dataset.build(
        batch_size=128, num_parallel_calls=8, repeat_count=1
    )
    result = kr_model.evaluate(val_dataset)
    print(dict(zip(kr_model.metrics_names, result)))


def get_kr_layer(kr_model, kr_layer_name):
    kr_layer = None
    for l in kr_model.layers:
        if l.name == kr_layer_name:
            kr_layer = l
            break
    return kr_layer


def compare_model_outputs(pt_model, kr_model):
    pt_layer_outputs = {}

    def get_pt_layer_outputs(name):
        def hook(model, input, output):
            pt_layer_outputs[name] = output.detach()

        return hook

    layer_pairs = [
        (pt_model.input.conv, get_kr_layer(kr_model, "input.conv")),
        (
            pt_model.sections[1][3].conv3,
            get_kr_layer(kr_model, "sections.1.3.conv3"),
        ),
        (
            pt_model.sections[3][2].conv3,
            get_kr_layer(kr_model, "sections.3.2.conv3"),
        ),
        (pt_model.classifier.softmax, get_kr_layer(kr_model, "classifier.fc")),
    ]
    pt_layer, kr_layer = layer_pairs[-1]

    pt_model.eval()
    pt_layer.register_forward_hook(get_pt_layer_outputs("pt_layer_output"))

    image = load_the_image()  # [1, 3, 224, 224]
    image_batch = torch.from_numpy(image)
    pt_model(image_batch)

    print("PT layer output:")
    pt_layer_outputs = pt_layer_outputs["pt_layer_output"]
    print(pt_layer_outputs.shape)
    print(pt_layer_outputs)

    kr_sub_model = keras.models.Model(inputs=kr_model.input, outputs=kr_layer.output)
    image_batch = np.transpose(image, [0, 2, 3, 1])  # [1, 224, 224, 3]
    assert image_batch.shape == (1, 224, 224, 3)

    kr_layer_outputs = kr_sub_model(image_batch, training=False)

    if len(kr_layer_outputs.shape) == 4:
        # For Conv
        kr_layer_outputs = np.transpose(
            kr_layer_outputs, [0, 3, 1, 2]
        )  # [1, 3, 224, 224]
    print("KR layer output:")
    print(kr_layer_outputs.shape)
    print(kr_layer_outputs)

    x = pt_layer_outputs.numpy()
    y = kr_layer_outputs
    diff = np.mean(np.absolute(x - y))
    print("Average diff: {}".format(diff))


def _convert_model(
    arch_key: str,
    model_type: str,
    class_type: str,
    layer_mapping: Dict[str, str],
):
    print("Loading PyTorch model {}, {}".format(arch_key, model_type))
    pt_model = PTModelRegistry.create(
        arch_key, pretrained=model_type, class_type=class_type
    )
    pt_model_state_dict = pt_model.state_dict()

    kr_model = KRModelRegistry.create(arch_key, class_type=class_type)
    kr_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
        run_eagerly=True,
    )
    for layer in kr_model.layers:
        pt_layer_name = get_pytorch_name(layer.name, layer_mapping)
        if pt_layer_name is None:
            print(
                "W: {} layer in Keras will not be updated from Pytorch".format(
                    layer.name
                )
            )
            continue
        else:
            print(
                "{} layer in Keras updated from {} in Pytorch".format(
                    layer.name, pt_layer_name
                )
            )
        if isinstance(layer, keras.layers.Conv2D):
            convert_conv_layer(layer, pt_layer_name, pt_model_state_dict)
        elif isinstance(layer, keras.layers.BatchNormalization):
            convert_bn_layer(layer, pt_layer_name, pt_model_state_dict)
        elif isinstance(layer, keras.layers.Dense):
            convert_dense_layer(layer, pt_layer_name, pt_model_state_dict)

    compare_model_outputs(pt_model, kr_model)
    return kr_model


def convert_pytorch_to_keras(
    arch_key: str,
    type_: str,
    class_type: str,
    layer_mapping: Dict[str, str],
    output_dir: str,
    imagenet_dir: str,
):
    model_dir = os.path.join(output_dir, "{}-{}".format(arch_key, type_))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_file_path = os.path.join(model_dir, "model.h5")

    # Convert and save
    kr_model = _convert_model(arch_key, type_, class_type, layer_mapping)
    kr_model.save(model_file_path)

    # Export to ONNX
    exporter = ModelExporter(kr_model, model_dir)
    exporter.export_onnx(name="model.onnx", debug_mode=False)

    # Samples
    samples_dir = os.path.join(model_dir, "samples")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    val_dataset = ImageNetDataset(IMAGENET_DIR, train=False)
    val_dataset = val_dataset.build(batch_size=100, repeat_count=1)
    for img_batch, label_batch in val_dataset.take(1):
        output_batch = kr_model(img_batch)
        np.save(os.path.join(samples_dir, "inputs.npy"), img_batch)
        np.save(os.path.join(samples_dir, "outputs.npy"), output_batch)
        np.save(os.path.join(samples_dir, "labels.npy"), label_batch)


def convert_resnets_for_keras(output_dir: str):
    resnet_mapping = {
        "input.conv": "input.conv",
        "input.bn": "input.bn",
        "sections.([0-9]+).([0-9]+).conv([0-9]+)": "sections.{}.{}.conv{}",
        "sections.([0-9]+).([0-9]+).bn([0-9]+)": "sections.{}.{}.bn{}",
        "sections.([0-9]+).([0-9]+).identity.conv": "sections.{}.{}.identity.conv",
        "sections.([0-9]+).([0-9]+).identity.bn": "sections.{}.{}.identity.bn",
        "classifier.fc": "classifier.fc",
    }
    convert_pytorch_to_keras(
        "resnet50",
        "base-none",
        "single",
        resnet_mapping,
        output_dir,
        IMAGENET_DIR,
    )
    convert_pytorch_to_keras(
        "resnet50",
        "pruned-conservative",
        "single",
        resnet_mapping,
        output_dir,
        IMAGENET_DIR,
    )
    convert_pytorch_to_keras(
        "resnet50",
        "pruned-moderate",
        "single",
        resnet_mapping,
        output_dir,
        IMAGENET_DIR,
    )
    convert_pytorch_to_keras(
        "resnet101",
        "base-none",
        "single",
        resnet_mapping,
        output_dir,
        IMAGENET_DIR,
    )
    convert_pytorch_to_keras(
        "resnet101",
        "pruned-moderate",
        "single",
        resnet_mapping,
        output_dir,
        IMAGENET_DIR,
    )
    convert_pytorch_to_keras(
        "resnet152",
        "base-none",
        "single",
        resnet_mapping,
        output_dir,
        IMAGENET_DIR,
    )
    convert_pytorch_to_keras(
        "resnet152",
        "pruned-moderate",
        "single",
        resnet_mapping,
        output_dir,
        IMAGENET_DIR,
    )


def main():
    convert_resnets_for_keras(OUTPUT_DIR)


if __name__ == "__main__":
    main()
