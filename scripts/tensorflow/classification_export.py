"""
Image classification export script. Exports models to a standard structure
including an ONNX export as well as sample inputs, outputs, and labels


##########
Command help:
python scripts/tensorflow/classification_export.py -h
usage: classification_export.py [-h] [--num-samples NUM_SAMPLES] --arch-key
                                ARCH_KEY [--pretrained PRETRAINED]
                                [--pretrained-dataset PRETRAINED_DATASET]
                                [--checkpoint-path CHECKPOINT_PATH]
                                [--class-type CLASS_TYPE] --dataset DATASET
                                --dataset-path DATASET_PATH
                                [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]
                                [--onnx-opset ONNX_OPSET]

Export an image classification model to onnx as well as store sample inputs,
outputs, and labels

optional arguments:
  -h, --help            show this help message and exit
  --num-samples NUM_SAMPLES
                        The number of samples to export along with the model
                        onnx and pth files (sample inputs and labels as well
                        as the outputs from model execution)
  --arch-key ARCH_KEY   The type of model to create, ex: resnet50, vgg16,
                        mobilenet put as help to see the full list (will raise
                        an exception with the list)
  --pretrained PRETRAINED
                        The type of pretrained weights to use, default is true
                        to load the default pretrained weights for the model.
                        Otherwise should be set to the desired weights type:
                        [base, recal, recal-perf]. To not load any weights set
                        to one of [none, false]
  --pretrained-dataset PRETRAINED_DATASET
                        The dataset to load pretrained weights for if
                        pretrained is set. Default is None which will load the
                        default dataset for the architecture. Ex can be set to
                        imagenet, cifar10, etc
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained will
                        be ignored
  --class-type CLASS_TYPE
                        One of [single, multi] where single is for single
                        class training using a softmax and multi is for multi
                        class training using a sigmoid
  --dataset DATASET     The dataset to use for training, ex: imagenet,
                        imagenette, cifar10, etc. Set to imagefolder for a
                        generic image classification dataset setup with an
                        image folder structure setup like imagenet.
  --dataset-path DATASET_PATH
                        The root path to where the dataset is stored
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir, defaults to the model arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results
  --onnx-opset ONNX_OPSET
                        The onnx opset to use for export. Default is 11


##########
Example command for exporting ResNet50:
python scripts/tensorflow/classification_export.py \
    --arch-key resnet50 --dataset imagenet --dataset-path ~/datasets/ILSVRC2012
"""

import argparse
import os
from tqdm import auto

from neuralmagicML import get_main_logger
from neuralmagicML.tensorflow.datasets import DatasetRegistry
from neuralmagicML.tensorflow.models import ModelRegistry
from neuralmagicML.tensorflow.utils import GraphExporter, tf_compat
from neuralmagicML.utils import create_dirs


LOGGER = get_main_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export an image classification model to onnx as well as "
        "store sample inputs, outputs, and labels"
    )

    # recal
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="The number of samples to export along with the model onnx and pth files "
        "(sample inputs and labels as well as the outputs from model execution)",
    )

    # model args
    parser.add_argument(
        "--arch-key",
        type=str,
        required=True,
        help="The type of model to create, ex: resnet50, vgg16, mobilenet "
        "put as help to see the full list (will raise an exception with the list)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=True,
        help="The type of pretrained weights to use, "
        "default is true to load the default pretrained weights for the model. "
        "Otherwise should be set to the desired weights type: "
        "[base, recal, recal-perf]. "
        "To not load any weights set to one of [none, false]",
    )
    parser.add_argument(
        "--pretrained-dataset",
        type=str,
        default=None,
        help="The dataset to load pretrained weights for if pretrained is set. "
        "Default is None which will load the default dataset for the architecture."
        " Ex can be set to imagenet, cifar10, etc",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="A path to a previous checkpoint to load. "
        "If provided, pretrained will be ignored",
    )
    parser.add_argument(
        "--class-type",
        type=str,
        default="single",
        help="One of [single, multi] where single is for single class training "
        "using a softmax and multi is for multi class training using a sigmoid",
    )

    # dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to use for training, "
        "ex: imagenet, imagenette, cifar10, etc. "
        "Set to imagefolder for a generic image classification dataset setup "
        "with an image folder structure setup like imagenet.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="The root path to where the dataset is stored",
    )

    # logging and saving
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="A tag to use for the model for saving results under save-dir, "
        "defaults to the model arch and dataset used",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="tensorflow_classification_export",
        help="The path to the directory for saving results",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=11,
        help="The onnx opset to use for export. Default is 11",
    )

    return parser.parse_args()


def main(args):
    # logging and saving setup
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))

    if not args.model_tag:
        model_tag = "{}_{}".format(args.arch_key.replace("/", "."), args.dataset)
        model_id = model_tag
        model_inc = 0

        while os.path.exists(os.path.join(args.save_dir, model_id)):
            model_inc += 1
            model_id = "{}__{:02d}".format(model_tag, model_inc)
    else:
        model_id = args.model_tag

    save_dir = os.path.join(save_dir, model_id)
    create_dirs(save_dir)

    # loggers setup
    LOGGER.info("Model id is set to {}".format(model_id))

    # dataset creation
    input_shape = ModelRegistry.input_shape(args.arch_key)

    val_dataset = DatasetRegistry.create(
        args.dataset, root=args.dataset_path, train=False
    )

    LOGGER.info("created val_dataset: {}".format(val_dataset))

    # model creation
    if args.dataset == "imagefolder":
        num_classes = val_dataset.num_classes
    else:
        dataset_attributes = DatasetRegistry.attributes(args.dataset)
        num_classes = dataset_attributes["num_classes"]

    with tf_compat.Graph().as_default() as graph:
        inputs = tf_compat.placeholder(
            tf_compat.float32, [None] + list(input_shape), name="inputs"
        )
        outputs = ModelRegistry.create(
            args.arch_key,
            inputs,
            training=False,
            num_classes=num_classes,
            class_type=args.class_type,
        )
        with tf_compat.Session() as sess:
            ModelRegistry.load_pretrained(
                args.arch_key,
                pretrained=args.pretrained,
                pretrained_dataset=args.pretrained_dataset,
                pretrained_path=args.checkpoint_path,
                sess=sess,
            )
            LOGGER.info("created model: {}".format(model_id))

            exporter = GraphExporter(save_dir)

            # Export a batch of samples and expected outputs
            tf_dataset = val_dataset.build(
                args.num_samples, repeat_count=1, num_parallel_calls=1
            )
            tf_iter = tf_compat.data.make_one_shot_iterator(tf_dataset)
            features, _ = tf_iter.get_next()
            inputs_val = sess.run(features)
            exporter.export_samples([inputs], [inputs_val], [outputs], sess)

            # Export model to tensorflow checkpoint format
            LOGGER.info("exporting tensorflow in {}".format(save_dir))
            exporter.export_checkpoint(sess=sess)

            # Export model to pb format
            LOGGER.info("exporting pb in {}".format(exporter.pb_path))
            exporter.export_pb(outputs=[outputs])

    # Export model to onnx format
    LOGGER.info("exporting onnx in {}".format(exporter.onnx_path))
    exporter.export_onnx([inputs], [outputs], opset=args.onnx_opset)


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
