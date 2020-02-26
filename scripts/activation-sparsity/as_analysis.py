from typing import Union, List, Dict
import argparse
import os
import json
import csv
import numpy
import matplotlib.pyplot as plt
import pandas
import torch
from torch.utils.data import DataLoader
from torch.nn import Linear, Module, ReLU
from torch.nn.modules.conv import _ConvNd

from neuralmagicML.pytorch.datasets import create_dataset, EarlyStopDataset
from neuralmagicML.pytorch.models import create_model, model_to_device
from neuralmagicML.pytorch.recal import ModuleASAnalyzer, FATReLU
from neuralmagicML.pytorch.utils import (
    CrossEntropyLossWrapper,
    ModuleTester,
    convert_to_bool,
)


def as_analysis(
    device_desc: str,
    model_type: str,
    pretrained: Union[bool, str],
    model_path: Union[None, str],
    model_plugin_paths: Union[None, List[str]],
    model_plugin_args: Union[None, Dict],
    dataset_type: str,
    dataset_root: str,
    sample_size: int,
    test_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    save_dir: str,
):

    ####################################################################################################################
    #
    # Dataset and data loader setup section
    #
    ####################################################################################################################

    test_dataset, num_classes = create_dataset(
        dataset_type, dataset_root, train=False, rand_trans=False
    )
    test_dataset = EarlyStopDataset(test_dataset, early_stop=sample_size)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print("created test dataset: \n{}\n".format(test_dataset))

    ####################################################################################################################
    #
    # Model, knowledge distillation, and loss function setup
    #
    ####################################################################################################################

    model = create_model(
        model_type,
        model_path,
        plugin_paths=model_plugin_paths,
        plugin_args=model_plugin_args,
        pretrained=pretrained,
        num_classes=num_classes,
    )  # type: Module
    print(
        "Created model of type {} with num_classes:{}, pretrained:{} / model_path:{}, and plugins:{}".format(
            model_type, num_classes, pretrained, model_path, model_plugin_paths
        )
    )

    model, device, device_ids = model_to_device(model, device_desc)

    print("Transferred model(s) to device {} with ids {}".format(device, device_ids))

    save_dir = os.path.abspath(os.path.expanduser(save_dir))
    model_tag = "{}-{}".format(model_type.replace("/", "."), dataset_type)
    model_id = model_tag
    model_inc = 0

    while os.path.exists(os.path.join(save_dir, model_id)):
        model_inc += 1
        model_id = "{}__{:02d}".format(model_tag, model_inc)

    print("Model id is set to {}".format(model_id))
    model_dir = os.path.join(save_dir, model_id)
    os.makedirs(model_dir)
    print("Saving results to {}".format(model_dir))

    ####################################################################################################################
    #
    # Activation Sparsity and testing setup
    #
    ####################################################################################################################

    tester = ModuleTester(model, device, CrossEntropyLossWrapper())
    as_analyzer_layers = {}

    for name, mod in model.named_modules():
        if isinstance(mod, _ConvNd) or isinstance(mod, Linear):
            as_analyzer_layers[name] = ModuleASAnalyzer(
                mod, division=0, track_inputs_sparsity=True
            )
        elif isinstance(mod, ReLU) or isinstance(mod, FATReLU):
            as_analyzer_layers[name] = ModuleASAnalyzer(
                mod, division=0, track_outputs_sparsity=True
            )

    ####################################################################################################################
    #
    # Analyzing AS and saving results
    #
    ####################################################################################################################

    print("Testing activation sparsity")
    for name, analyzer in as_analyzer_layers.items():
        analyzer.enable()

    tester.run(
        test_dataloader, desc="AS tracking...", show_progress=True, track_results=False
    )

    for name, analyzer in as_analyzer_layers.items():
        analyzer.disable()

    layer_results = {"relus": {}, "convs": {}}

    for name, layer in as_analyzer_layers.items():
        if layer.track_inputs_sparsity:
            layer_results["convs"][name] = (
                torch.cat(layer.inputs_sparsity).view(-1).tolist()
            )
        elif layer.track_outputs_sparsity:
            layer_results["relus"][name] = (
                torch.cat(layer.outputs_sparsity).view(-1).tolist()
            )
        else:
            raise RuntimeError("unknown type of layer analyzer")

    json_path = os.path.join(model_dir, "activation-sparsity.json")
    with open(json_path, "w") as json_file:
        json.dump(layer_results, json_file)
    print("saved json dump to {}".format(json_path))

    csv_path = os.path.join(model_dir, "activation-sparsity.csv")
    with open(csv_path, "w") as csv_file:
        csv_writer = csv.writer(
            csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writerow(["Layer Type", "Layer Name", "AS mean", "AS stddev"])

        for relu_name, relu_sparsities in layer_results["relus"].items():
            csv_writer.writerow(
                [
                    "relu",
                    relu_name,
                    numpy.mean(relu_sparsities),
                    numpy.std(relu_sparsities),
                ]
            )

        for conv_name, conv_sparsities in layer_results["convs"].items():
            csv_writer.writerow(
                [
                    "conv",
                    conv_name,
                    numpy.mean(conv_sparsities),
                    numpy.std(conv_sparsities),
                ]
            )
    print("saved csv summary to {}".format(csv_path))

    as_data = {}
    relu_counter = 0

    for relu_name, relu_sparsities in layer_results["relus"].items():
        plot_name = "relu{:04d}  {}".format(relu_counter, relu_name)
        relu_counter += 1
        as_data[plot_name] = numpy.mean(relu_sparsities)

    conv_counter = 0

    for conv_name, conv_sparsities in layer_results["convs"].items():
        plot_name = "conv{:04d}  {}".format(conv_counter, conv_name)
        conv_counter += 1
        as_data[plot_name] = numpy.mean(conv_sparsities)

    height = round((len(layer_results["relus"]) + len(layer_results["convs"])) / 4) + 3
    fig = plt.figure(figsize=(10, height))
    ax = fig.add_subplot(111)
    ax.set_title("Activation Sparsity")
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Layer")
    plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)
    frame = pandas.DataFrame(data={"as": as_data})
    frame.plot.barh(ax=ax)
    plt_path = os.path.join(model_dir, "activation-sparsity.png")
    plt.savefig(plt_path)
    plt.close(fig)
    print("saved plot to {}".format(plt_path))

    print("Completed")


def main():
    parser = argparse.ArgumentParser(
        description="Train a model for a given dataset with a given schedule"
    )

    # schedule device and model arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cpu" if not torch.cuda.is_available() else "cuda",
        help="The device to run on (can also include ids for data parallel), ex: "
        "cpu, cuda, cuda:0,1",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="The type of model to create, ex: resnet/50, vgg/16, mobilenet/1.0 "
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

    # dataset settings
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        help="The dataset type to load for training, ex: imagenet, imagenette, cifar10, etc",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="The root path to where the dataset is stored",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        required=True,
        help="The number of samples to run through for tracking activation sparsity",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        required=True,
        help="The batch size to use while testing",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="The number of workers to use for data loading",
    )
    parser.add_argument(
        "--pin-memory",
        type=bool,
        default=True,
        help="Use pinned memory for data loading",
    )

    # save options
    parser.add_argument(
        "--save-dir",
        type=str,
        default="as-analysis",
        help="The path to the directory for saving results",
    )

    args = parser.parse_args()
    as_analysis(
        args.device,
        args.model_type,
        args.pretrained,
        args.model_path,
        args.model_plugin_paths,
        args.model_plugin_args,
        args.dataset_type,
        args.dataset_root,
        args.sample_size,
        args.test_batch_size,
        args.num_workers,
        convert_to_bool(args.pin_memory),
        args.save_dir,
    )


if __name__ == "__main__":
    main()
