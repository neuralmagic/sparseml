from typing import Union, List, Dict
import argparse
import os
import json

from torch.utils.data import DataLoader
import torchvision.models as classificiation_models
import torchvision.models.segmentation as segmentation_models
import torchvision.models.detection as detection_models
import torchvision.models.video as video_models

# from neuralmagicML.pytorch.datasets -import ImageNetDataset
# from neuralmagicML.pytorch.datasets import MNISTDataset
from neuralmagicML.pytorch.datasets import DatasetRegistry
from neuralmagicML.pytorch.models import ModelRegistry
from neuralmagicML.pytorch.utils import ModuleExporter


def export_model(
    model_type: str,
    pretrained: Union[bool, str],
    model_path: Union[None, str],
    num_classes: int,
    export_dir: Union[None, str],
    export_batch_sizes: str,
    export_input_sizes: str,
    export_intermediates: bool,
    export_layers: bool,
    model_plugin_paths: Union[None, List[str]],
    model_plugin_args: Union[None, Dict],
    dataset_path: str,
    dataset_type: str,
    from_zoo: bool,
    zoo_model_domain: str,
):
    # fix for exporting FATReLU's
    # fix_onnx_threshold_export()

    if export_dir is None:
        export_dir = os.path.join(".", "onnx", model_type)

    export_dir = os.path.abspath(os.path.expanduser(export_dir))

    if from_zoo:
        if zoo_model_domain == "classification":
            models = classificiation_models
        elif zoo_model_domain == "segmentation":
            models = segmentation_models
        elif zoo_model_domain == "detection":
            models = detection_models
        elif zoo_model_domain == "video":
            models = video_models
        else:
            raise Exception("Not using a valid zoo model domain type")
        model = getattr(models, model_type)(pretrained=True)
    else:
        model = ModelRegistry.create(
            key=model_type, pretrained=pretrained, pretrained_path=model_path
        )
    print(
        "Created model of type {} with num_classes:{}, pretrained:{} / model_path:{}, and plugins:{}".format(
            model_type, num_classes, pretrained, model_path, model_plugin_paths
        )
    )

    export_batch_sizes = [
        int(batch_size) for batch_size in export_batch_sizes.split(",")
    ]
    export_input_sizes = [
        tuple([int(size) for size in input_size.split(",")])
        for input_size in export_input_sizes.split(";")
    ]

    print(
        "Exporting model to {} for input_sizes:{}, batch_sizes:{}, intermediates:{}, and layers:{}".format(
            export_dir,
            export_batch_sizes,
            export_input_sizes,
            export_intermediates,
            export_layers,
        )
    )

    if dataset_type == "imagenet":
        dataset = ImageNetDataset(root=dataset_path)
    elif dataset_type == "mnist":
        dataset = MNISTDataset(root=dataset_path)
    else:
        raise ValueError(f"Unsupported dataset type {dataset_type}")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    data_iter = iter(dataloader)
    samples = [next(data_iter)[0] for i in range(100)]

    exporter = ModuleExporter(model, export_dir)
    exporter.export_onnx(sample_batch=samples[1])
    exporter.export_pytorch()
    exporter.export_samples(samples)

    print("Completed")


def main():
    parser = argparse.ArgumentParser(
        description="Export a model for a given dataset with "
    )

    # schedule device and model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="The type of model to create, ex: resnet50, vgg16, mobilenet "
        "put as help to see the full list (will raise an exception with the list)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="The type of pretrained weights to use, default is not to load any, "
        "ex: imagenet/dense, imagenet/sparse, imagenette/dense",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="The path to the model file to load a previous state dict from",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=False,
        help="The number of classes to create the model for",
    )

    # export settings
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="The directory to export the model under, defaults to onnx/MODEL_TYPE",
    )
    parser.add_argument(
        "--export-batch-sizes",
        type=str,
        default="1",
        help="A comma separated list of the batch sizes to export, ex: 1,16,64",
    )
    parser.add_argument(
        "--export-input-sizes",
        type=str,
        default="3,224,224",
        help="A comma and semi colon separated list for the inputs to use to the model "
        "commas separate the shapes within a single input, semi colon separates multi input "
        "ex: 3,224,224   3,224,224;3,128,128",
    )
    parser.add_argument(
        "--export-intermediates",
        type=bool,
        default=False,
        help="Export the intermediate tensors within the model execution",
    )
    parser.add_argument(
        "--export-layers",
        type=bool,
        default=False,
        help="Export the layer params for the model to numpy files",
    )

    # plugin options
    parser.add_argument(
        "--model-plugin-paths",
        default=None,
        nargs="+",
        help="plugins to load for handling a model and possibly teacher after creation",
    )
    parser.add_argument(
        "--model-plugin-args",
        type=json.loads,
        default={},
        help="json string containing the args to pass to the model plugins when executing",
    )

    parser.add_argument(
        "--dataset-path", type=str, required=True, help="The path to dataset",
    )

    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        help="The type of dataset the model was trained with",
    )

    parser.add_argument(
        "--from-zoo", action="store_true", help="Download model from zoo"
    )

    parser.add_argument(
        "--zoo-model-domain",
        default="classification",
        type=str,
        help="The domain of zoo model, defaults to classification",
    )

    args = parser.parse_args()
    export_model(
        args.model_type,
        args.pretrained,
        args.model_path,
        args.num_classes,
        args.export_dir,
        args.export_batch_sizes,
        args.export_input_sizes,
        args.export_intermediates,
        args.export_layers,
        args.model_plugin_paths,
        args.model_plugin_args,
        args.dataset_path,
        args.dataset_type,
        args.from_zoo,
        args.zoo_model_domain,
    )


if __name__ == "__main__":
    main()
