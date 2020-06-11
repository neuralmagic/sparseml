import argparse

from neuralmagicML.utils import RepoModel, available_models


DOWNLOAD_COMMAND = "download"
AVAILABLE_COMMAND = "available"


def parse_args():
    parser = argparse.ArgumentParser(description="Download deployed models")

    subparsers = parser.add_subparsers(dest="command")
    download_parser = subparsers.add_parser(DOWNLOAD_COMMAND)
    available_parser = subparsers.add_parser(AVAILABLE_COMMAND)

    download_parser.add_argument(
        "--target-path",
        type=str,
        default="models/",
        required=True,
        help="the target path for the model folder to be saved",
    )

    download_parser.add_argument(
        "--download-test-data",
        action="store_true",
        help="downloads test input/output/label data",
    )

    for subparser, required in [(download_parser, True), (available_parser, False)]:

        subparser.add_argument(
            "--dom",
            type=str,
            required=required,
            help="the domain the model belongs to; ex: cv, nlp, etc",
        )
        subparser.add_argument(
            "--sub-dom",
            type=str,
            required=required,
            help="the sub domain the model belongs to; ex: classification, detection, etc",
        )
        subparser.add_argument(
            "--arch",
            type=str,
            required=required,
            help="the architecture the model belongs to; ex: resnet, mobilenet, etc",
        )
        subparser.add_argument(
            "--sub-arch",
            type=str,
            default="none",
            help="the sub architecture the model belongs to; ex: 50, 101, etc",
        )
        subparser.add_argument(
            "--dataset",
            type=str,
            required=required,
            help="the dataset used to train the model; ex: imagenet, cifar, etc",
        )
        subparser.add_argument(
            "--framework",
            type=str,
            required=required,
            help="the framework used to train the model; ex: tensorflow, pytorch, keras, onnx, etc",
        )
        subparser.add_argument(
            "--desc",
            type=str,
            required=required,
            help="the description of the model; ex: base, recal, recal_perf",
        )

    return parser.parse_args()


def main():
    args = parse_args()

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

        model.download_onnx_file(overwrite=False, save_dir=args.model_path)
        model.download_framework_files(overwrite=False, save_dir=args.model_path)
        if args.download_test_data:
            model.download_data_files(overwrite=False, save_dir=args.model_path)
    else:
        models = available_models(
            domains=args.dom,
            sub_domains=args.sub_dom,
            architectures=args.arch,
            sub_architectures=args.sub_arch,
            datasets=args.dataset,
            frameworks=args.framework,
            descs=args.desc,
        )
        print("Available Models:")
        for model in models:
            print(model)


if __name__ == "__main__":
    main()
