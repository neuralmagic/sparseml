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

"""
Script to convert Pytorch models to Keras

usage: pytorch2keras.py [-h] --arch-key ARCH_KEY --model-type MODEL_TYPE
                        --imagenet-dir IMAGENET_DIR --test-image-file-path
                        TEST_IMAGE_FILE_PATH --save-dir SAVE_DIR

Convert Pytorch models to Keras

optional arguments:
  -h, --help            show this help message and exit
  --arch-key ARCH_KEY   Arch key of model to convert (e.g., 'resnet50')
  --model-type MODEL_TYPE
                        Type of model to convert (e.g., 'base-none', 'pruned-
                        moderate')
  --imagenet-dir IMAGENET_DIR
                        The root path to where the Imagenet dataset is stored
  --test-image-file-path TEST_IMAGE_FILE_PATH
                        Path to an image used for comparing inference results
                        between Pytorch and Keras
  --save-dir SAVE_DIR   The path to the directory for saving results

############
EXAMPLE:

python utils/pytorch2keras.py \
    --arch-key resnet101 \
    --model-type pruned-moderate \
    --imagenet-dir /hdd/datasets/ILSVRC \
    --save-dir /hdd/src/sparseml/pytorch2keras_models/ \
    --test-image-file-path \
        /hdd/datasets/ILSVRC/val/n02481823/ILSVRC2012_val_00024600.JPEG
"""
import argparse
import os
import re
from typing import Dict

import numpy as np
import tensorflow
import torch
import torchvision.transforms as transforms
from PIL import Image

from sparseml.keras.datasets import ImageNetDataset, SplitsTransforms
from sparseml.keras.models import ModelRegistry as KRModelRegistry
from sparseml.keras.utils import ModelExporter, keras
from sparseml.pytorch.models import ModelRegistry as PTModelRegistry
from sparseml.utils.datasets import IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS


BN_MOMENTUM = 0.9
BN_EPSILON = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Pytorch models to Keras")
    parser.add_argument(
        "--arch-key",
        type=str,
        required=True,
        help="Arch key of model to convert (e.g., 'resnet50')",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="Type of model to convert (e.g., 'base-none', 'pruned-moderate')",
    )
    parser.add_argument(
        "--imagenet-dir",
        type=str,
        required=True,
        help="The root path to where the Imagenet dataset is stored",
    )
    parser.add_argument(
        "--test-image-file-path",
        type=str,
        required=True,
        help="Path to an image used for comparing inference results between "
        "Pytorch and Keras",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="The path to the directory for saving results",
    )
    return parser.parse_args()


def load_the_image(test_image_file_path):
    image = Image.open(test_image_file_path)
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


def verify_keras_model(kr_model, imagenet_dir):
    batch_size = 128
    val_dataset = ImageNetDataset(imagenet_dir, train=False)
    val_dataset = val_dataset.build(
        batch_size=batch_size, num_parallel_calls=8, repeat_count=1
    )
    result = kr_model.evaluate(val_dataset)
    print(dict(zip(kr_model.metrics_names, result)))


def verify_keras_model_with_pytorch(kr_model, imagenet_dir):
    """
    Verify the converted models using ImageNet's data pipeline in Pytorch
    Assumption: the validation pipeline is enhanced with the following permutation
    class my_permuter:
        def __call__(self, img):
            return img.permute(1, 2, 0)
    to fit into the default data format "channels_last" by Keras
    """
    from torch.utils.data import DataLoader

    from sparseml.pytorch.datasets import ImageNetDataset as PTImageNetDataset

    batch_size = 128
    val_dataset = PTImageNetDataset(imagenet_dir, train=False)

    val_loader = DataLoader(
        val_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=8
    )
    val_acc_metric = keras.metrics.CategoricalAccuracy()
    for x_batch_val, y_batch_val in val_loader:
        val_logits = kr_model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(
            torch.nn.functional.one_hot(y_batch_val, num_classes=1000), val_logits
        )
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))


def get_kr_layer(kr_model, kr_layer_name):
    kr_layer = None
    for layer in kr_model.layers:
        if layer.name == kr_layer_name:
            kr_layer = layer
            break
    return kr_layer


def compare_model_outputs(pt_model, kr_model, test_image_file_path):
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

    image = load_the_image(test_image_file_path)  # [1, 3, 224, 224]
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
    test_image_file_path: str,
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

    compare_model_outputs(pt_model, kr_model, test_image_file_path)
    return kr_model


def convert_pytorch_to_keras(
    arch_key: str,
    type_: str,
    class_type: str,
    layer_mapping: Dict[str, str],
    output_dir: str,
    imagenet_dir: str,
    test_image_file_path: str,
):
    model_dir = os.path.join(output_dir, "{}-{}".format(arch_key, type_))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_file_path = os.path.join(model_dir, "model.h5")

    # Convert and save
    kr_model = _convert_model(
        arch_key, type_, class_type, layer_mapping, test_image_file_path
    )
    kr_model.save(model_file_path)

    verify_keras_model(kr_model, imagenet_dir)
    # verify_keras_model_with_pytorch(kr_model, imagenet_dir)

    # Export to ONNX
    exporter = ModelExporter(kr_model, model_dir)
    exporter.export_onnx(name="model.onnx", debug_mode=False)

    # Samples
    n_samples = 20
    samples_dir = os.path.join(model_dir, "samples")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    val_dataset = ImageNetDataset(imagenet_dir, train=False)
    val_dataset = val_dataset.build(
        batch_size=n_samples, shuffle_buffer_size=None, repeat_count=1
    )
    for img_batch, label_batch in val_dataset.take(1):
        output_batch = kr_model(img_batch)
        np.save(os.path.join(samples_dir, "inputs.npy"), img_batch)
        np.save(os.path.join(samples_dir, "outputs.npy"), output_batch)
        np.save(os.path.join(samples_dir, "labels.npy"), label_batch)

    def zero_padding_image():
        def _zero_padding_image(image):
            max_image_size = 1024
            return tensorflow.image.pad_to_bounding_box(
                image, 1, 1, max_image_size, max_image_size
            )

        return _zero_padding_image

    pad_val_dataset = ImageNetDataset(
        imagenet_dir,
        image_size=None,
        train=False,
        pre_resize_transforms=SplitsTransforms(train=None, val=(zero_padding_image(),)),
        post_resize_transforms=SplitsTransforms(train=None, val=None),
    )
    pad_val_dataset = pad_val_dataset.build(
        batch_size=n_samples, shuffle_buffer_size=None, repeat_count=1
    )
    for padded_img_batch, _ in pad_val_dataset.take(1):
        np.save(os.path.join(samples_dir, "originals.npy"), padded_img_batch)


def convert_resnets_for_keras(args):
    output_dir = args.save_dir
    imagenet_dir = args.imagenet_dir
    test_image_file_path = args.test_image_file_path
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
        args.arch_key,
        args.model_type,
        "single",
        resnet_mapping,
        output_dir,
        imagenet_dir,
        test_image_file_path,
    )


def main(args):
    convert_resnets_for_keras(args)


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
