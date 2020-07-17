"""
Script to download an ONNX model from the Model Repo


##########
Command help:
usage: model_download.py [-h] {list,download} ...

Download deployed models

positional arguments:
  {list,download}

optional arguments:
  -h, --help       show this help message and exit


##########
list command help:
usage: model_download.py list [-h]

List out all available models in the repo.

optional arguments:
  -h, --help  show this help message and exit


##########
download command help:
usage: model_download.py download [-h] --dom DOM --sub-dom SUB_DOM --arch ARCH
                                  [--sub-arch SUB_ARCH] --dataset DATASET
                                  --framework FRAMEWORK --desc DESC
                                  [--save-dir SAVE_DIR]

Download a specific model from the repo.

optional arguments:
  -h, --help            show this help message and exit
  --dom DOM             the domain the model belongs to; ex: cv, nlp, etc
  --sub-dom SUB_DOM     the sub domain the model belongs to; ex:
                        classification, detection, etc
  --arch ARCH           the architecture the model belongs to; ex: resnet-v1,
                        mobilenet-v1, etc
  --sub-arch SUB_ARCH   the sub architecture the model belongs to; ex: 50,
                        101, etc
  --dataset DATASET     the dataset used to train the model; ex: imagenet,
                        cifar, etc
  --framework FRAMEWORK
                        the framework used to train the model; ex: tensorflow,
                        pytorch, keras, onnx, etc
  --desc DESC           the description of the model; ex: base, recal, recal-
                        perf
  --save-dir SAVE_DIR   The directory to save the model files in, defaults to
                        the cwd with the model description as a sub folder


##########
Example list:
python scripts/onnx/model_download.py list


##########
Example download resnet 50:
python scripts/onnx/model_download.py download \
    --dom cv --sub-dom classification --arch resnet-v1 --sub-arch 50 \
    --dataset imagenet --framework pytorch --desc recal


##########
Example download mobilenet v1:
python scripts/onnx/model_download.py download \
    --dom cv --sub-dom classification --arch mobilenet-v1 --sub-arch 1.0 \
    --dataset imagenet --framework pytorch --desc recal
"""

import argparse
import logging
import os

from neuralmagicML.utils import RepoModel, available_models


DOWNLOAD_COMMAND = "download"
LIST_COMMAND = "list"


def parse_args():
    parser = argparse.ArgumentParser(description="Download deployed models")

    subparsers = parser.add_subparsers(dest="command")
    list_parser = subparsers.add_parser(
        LIST_COMMAND, description="List out all available models in the repo.",
    )
    download_parser = subparsers.add_parser(
        DOWNLOAD_COMMAND, description="Download a specific model from the repo.",
    )
    download_parser.add_argument(
        "--dom",
        type=str,
        required=True,
        help="the domain the model belongs to; ex: cv, nlp, etc",
    )
    download_parser.add_argument(
        "--sub-dom",
        type=str,
        required=True,
        help="the sub domain the model belongs to; "
        "ex: classification, detection, etc",
    )
    download_parser.add_argument(
        "--arch",
        type=str,
        required=True,
        help="the architecture the model belongs to; ex: resnet-v1, mobilenet-v1, etc",
    )
    download_parser.add_argument(
        "--sub-arch",
        type=str,
        default="none",
        help="the sub architecture the model belongs to; ex: 50, 101, etc",
    )
    download_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="the dataset used to train the model; ex: imagenet, cifar, etc",
    )
    download_parser.add_argument(
        "--framework",
        type=str,
        required=True,
        help="the framework used to train the model; "
        "ex: tensorflow, pytorch, keras, onnx, etc",
    )
    download_parser.add_argument(
        "--desc",
        type=str,
        required=True,
        help="the description of the model; ex: base, recal, recal-perf",
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
        print("loading available models...")
        models = available_models()

        print("Available Models:")
        for model in models:
            print(model)

        return

    if args.command == DOWNLOAD_COMMAND:
        model = RepoModel(
            domain=args.dom,
            sub_domain=args.sub_dom,
            architecture=args.arch,
            sub_architecture=args.sub_arch,
            dataset=args.dataset,
            framework=args.framework,
            desc=args.desc,
        )

        save_dir = (
            args.save_dir
            if args.save_dir is not None
            else os.path.join(
                ".",
                "{}_{}_{}_{}".format(
                    model.architecture,
                    model.sub_architecture,
                    model.dataset,
                    model.desc,
                ),
            )
        )

        print("downloading ONNX file...")
        onnx_path = model.download_onnx_file(overwrite=True, save_dir=save_dir)

        print("downloading {} files...".format(model.framework))
        framework_paths = model.download_framework_files(
            overwrite=True, save_dir=save_dir
        )

        try:
            print("downloading sample data files...")
            data_paths = model.download_data_files(overwrite=True, save_dir=save_dir)
        except Exception as err:
            logging.warning("Error when downloading sample data: {}".format(err))
            data_paths = None

        print("\n\n")
        print("completed download for {}".format(model))
        print("ONNX: {}".format(onnx_path))
        print("{}: {}".format(model.framework, framework_paths))
        print("data: {}".format(data_paths))

        return

    raise ValueError("unknown command given of {}".format(args.command))


if __name__ == "__main__":
    main(parse_args())
