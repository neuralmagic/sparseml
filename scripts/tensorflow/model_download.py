"""
Script to download a TensorFlow model from the TensorFlow ModelRegistry / Model Repo


##########
Command help:
usage: model_download.py [-h] {list,download} ...

Download a TensorFlow model from the ModelRepo. Use the download command for
downloading

positional arguments:
  {list,download}

optional arguments:
  -h, --help       show this help message and exit


##########
list command help:
usage: model_download.py list [-h]

List out all available model keys in the repo. Note, there are multiple name
versions for each model

optional arguments:
  -h, --help  show this help message and exit


##########
download command help:
usage: model_download.py download [-h] --model-key MODEL_KEY
                                  [--pretrained-type PRETRAINED_TYPE]
                                  [--pretrained-dataset PRETRAINED_DATASET]
                                  [--save-dir SAVE_DIR]

Download a specific model by key from the repo.

optional arguments:
  -h, --help            show this help message and exit
  --model-key MODEL_KEY
                        Key for what model to download; ex: resnet50
  --pretrained-type PRETRAINED_TYPE
                        The type of pretrained weights to download; ex: dense,
                        recal, recal-perf. The default used is specific to
                        each model, but generally are the dense weights.
  --pretrained-dataset PRETRAINED_DATASET
                        The dataset for the pretrained weights to download;
                        ex: imagenet. The default used is specific to each
                        model.
  --save-dir SAVE_DIR   The directory to save the model files in, defaults to
                        the cwd with the model description as a sub folder


##########
Example:
python scripts/tensorflow/model_download.py download \
    --model-key resnet50 \
    --pretrained-type recal-perf
"""

import argparse
import os

from neuralmagicML import get_main_logger
from neuralmagicML.tensorflow.models import ModelRegistry


LOGGER = get_main_logger()
DOWNLOAD_COMMAND = "download"
LIST_COMMAND = "list"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a TensorFlow model from the ModelRepo. "
        "Use the download command for downloading and list command to see all available"
    )

    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser(
        LIST_COMMAND,
        description="List out all available model keys in the repo. "
        "Note, there are multiple name versions for each model",
    )

    download_parser = subparsers.add_parser(
        DOWNLOAD_COMMAND, description="Download a specific model by key from the repo.",
    )
    download_parser.add_argument(
        "--model-key",
        type=str,
        required=True,
        help="Key for what model to download; ex: resnet50",
    )
    download_parser.add_argument(
        "--pretrained-type",
        type=str,
        default=True,
        help="The type of pretrained weights to download; "
        "ex: dense, recal, recal-perf. "
        "The default used is specific to each model, but generally are the "
        "dense weights.",
    )
    download_parser.add_argument(
        "--pretrained-dataset",
        type=str,
        default=None,
        help="The dataset for the pretrained weights to download; "
        "ex: imagenet. The default used is specific to each model.",
    )
    download_parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="The directory to save the model files in, "
        "defaults to the cwd with the model description as a sub folder",
    )

    return parser.parse_args()


def main(args):
    if args.command == LIST_COMMAND:
        # print out results instead of log so they can't be filtered
        print(
            "List of available_keys "
            "(note, there are multiple name versions for each model):"
        )
        print(ModelRegistry.available_keys())

        return

    if args.command == DOWNLOAD_COMMAND:
        repo_model = ModelRegistry.create_repo(
            key=args.model_key,
            pretrained=args.pretrained_type,
            pretrained_dataset=args.pretrained_dataset,
        )

        save_dir = (
            args.save_dir
            if args.save_dir is not None
            else os.path.join(
                ".",
                "{}_{}_{}_{}".format(
                    repo_model.architecture,
                    repo_model.sub_architecture,
                    repo_model.dataset,
                    repo_model.desc,
                ),
            )
        )

        LOGGER.info("downloading ONNX file...")
        onnx_path = repo_model.download_onnx_file(overwrite=True, save_dir=save_dir)

        LOGGER.info("downloading TensorFlow files...")
        tensorflow_paths = repo_model.download_framework_files(
            overwrite=True, save_dir=save_dir
        )

        try:
            LOGGER.info("downloading sample data files...")
            data_paths = repo_model.download_data_files(
                overwrite=True, save_dir=save_dir
            )
        except Exception as err:
            LOGGER.warning("Error when downloading sample data: {}".format(err))
            data_paths = None

        # print out results instead of log so they can't be filtered
        print("completed download for {}".format(args.model_key))
        print("ONNX: {}".format(onnx_path))
        print("TensorFlow: {}".format(tensorflow_paths))
        print("data: {}".format(data_paths))

        return

    raise ValueError("unknown command given of {}".format(args.command))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
