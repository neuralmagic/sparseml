from typing import List, Dict
import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy
import pandas
import torch
from torch.nn import Module, ReLU

from neuralmagicML.pytorch.datasets import (
    create_dataset,
    EarlyStopDataset,
    CacheableDataset,
)
from neuralmagicML.pytorch.models import create_model, save_model
from neuralmagicML.pytorch.recal import (
    ModuleASOneShootBooster,
    LayerBoostResults,
    FATReLU,
)
from neuralmagicML.pytorch.utils import (
    CrossEntropyLossWrapper,
    TopKAccuracy,
    convert_to_bool,
)


def as_staticbooster(
    layers: List[str],
    device: str,
    model_type: str,
    pretrained: str,
    model_path: str,
    dataset_type: str,
    dataset_root: str,
    batch_size: int,
    sample_size: int,
    max_target_metric_loss: float,
    metric_key: str,
    metric_increases: bool,
    precision: float,
    save_dir: str,
):
    ####################################################################################################################
    #
    # Dataset and data loader setup section
    #
    ####################################################################################################################
    dataset, num_classes = create_dataset(
        dataset_type, dataset_root, train=True, rand_trans=False
    )

    if sample_size > -1:
        dataset = EarlyStopDataset(dataset, sample_size)

    dataset = CacheableDataset(dataset)

    print("created dataset: \n{}\n".format(dataset))

    ####################################################################################################################
    #
    # Model and loss function setup
    #
    ####################################################################################################################

    model = create_model(
        model_type, model_path, pretrained=pretrained, num_classes=num_classes
    )
    print(
        "Created model of type {} with num_classes:{}, pretrained:{} / model_path:{}".format(
            model_type, num_classes, pretrained, model_path
        )
    )

    # only support for now is cross entropy with top1 and top5 accuracy
    loss = CrossEntropyLossWrapper(
        extras={"top1acc": TopKAccuracy(1), "top5acc": TopKAccuracy(5)}
    )
    print("Created loss {}".format(loss))

    ####################################################################################################################
    #
    # Boosting section
    #
    ####################################################################################################################

    booster = ModuleASOneShootBooster(
        model,
        device,
        dataset,
        batch_size,
        loss,
        data_loader_kwargs={"num_workers": 0, "pin_memory": True, "shuffle": False},
    )

    if not layers:
        layers = [
            key
            for key, mod in model.named_modules()
            if isinstance(mod, ReLU) or isinstance(mod, FATReLU)
        ]

    results = booster.run_layers(
        layers, max_target_metric_loss, metric_key, metric_increases, precision
    )

    ####################################################################################################################
    #
    # Results section
    #
    ####################################################################################################################

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
    model_path = os.path.join(model_dir, "boosted.pth")
    save_model(model_path, model)
    print("model saved to {}".format(model_path))

    _save_as_compare_data(results, model, model_dir, metric_key)

    print("Completed")


def _save_as_compare_data(
    results: Dict[str, LayerBoostResults], model: Module, save_dir: str, metric_key: str
):
    results_txt = _results_text(results, model)
    print(results_txt)

    with open(os.path.join(save_dir, "compare.txt"), "w") as compare_file:
        compare_file.write(results_txt)

    as_baseline = {}
    as_final = {}
    as_full_data = {}
    layer_counter = 0

    for name, _ in model.named_modules():
        if name in results:
            plot_name = "{:04d}  {}".format(layer_counter, name)
            layer_counter += 1
            as_baseline[plot_name] = results[name].baseline_as.item()
            as_final[plot_name] = results[name].boosted_as.item()
            as_full_data[name] = {
                "baseline_as": results[name].baseline_as.item(),
                "boosted_as": results[name].boosted_as.item(),
                "baseline_loss": results[name]
                .baseline_loss.result_mean(metric_key)
                .item(),
                "boosted_loss": results[name]
                .boosted_loss.result_mean(metric_key)
                .item(),
            }

    as_baseline["__overall__"] = numpy.mean(
        [sparsity for key, sparsity in as_baseline.items()]
    )
    as_final["__overall__"] = numpy.mean(
        [sparsity for key, sparsity in as_final.items()]
    )

    height = round(len(as_baseline.keys()) / 4) + 3
    fig = plt.figure(figsize=(10, height))
    ax = fig.add_subplot(111)
    ax.set_title("Boosted Activation Sparsity Comparison")
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("Layer")
    plt.subplots_adjust(left=0.3, bottom=0.1, right=0.95, top=0.9)
    frame = pandas.DataFrame(data={"baseline": as_baseline, "final": as_final})
    frame.plot.barh(ax=ax)
    plt.savefig(os.path.join(save_dir, "compare.png"))
    plt.close(fig)

    with open(os.path.join(save_dir, "compare.json"), "w") as compare_file:
        json.dump(as_full_data, compare_file)


def _results_text(results: Dict[str, LayerBoostResults], model: Module) -> str:
    txt = "\n\n#############################################"
    txt += "\nBoosting Results"
    txt += "\n\nLosses"

    module_results = results["__module__"]
    for loss in module_results.baseline_loss.results.keys():
        txt += "\n    {}: {:.4f} -> {:.4f}".format(
            loss,
            module_results.baseline_loss.result_mean(loss).item(),
            module_results.boosted_loss.result_mean(loss).item(),
        )

    txt += "\n\nLayers Sparsity"

    for name, _ in model.named_modules():
        if name in results:
            result = results[name]
            txt += "\n    {}: {:.4f} -> {:.4f}".format(
                result.name, result.baseline_as.item(), result.boosted_as.item()
            )
    txt += "\n\n"

    return txt


def main():
    parser = argparse.ArgumentParser(
        description="Boost the activation sparsity for a model without retraining"
    )

    # model arguments
    parser.add_argument(
        "--layers",
        type=str,
        default=[],
        nargs="+",
        help="list of the layers to boost, if none given will boost all",
    )
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
        "--batch-size",
        type=int,
        required=True,
        help="The batch size to use while testing",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        required=True,
        help="The total number of samples to run through for calculating losses and sparsity values",
    )

    # boost options
    parser.add_argument(
        "--max-target-metric-loss",
        type=float,
        default=0.1,
        help="Maximum allowed loss in the given target metric while boosting the threshold for a layer",
    )
    parser.add_argument(
        "--metric-key",
        type=str,
        default="top1acc",
        help="Metric to compare loss with while boosting the threshold for a layer",
    )
    parser.add_argument(
        "--metric-increases",
        type=bool,
        default=False,
        help="True if the target metric increases for worse loss ie cross entropy,"
        "otherwise false ie accuracy",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=0.001,
        help="The desired precision to search for a threshold until, "
        "ie will stop when the difference in compare thresholds is less than 0.001",
    )

    # save options
    parser.add_argument(
        "--save-dir",
        type=str,
        default="as-staticboosting",
        help="The path to the directory for saving results",
    )

    args = parser.parse_args()
    as_staticbooster(
        args.layers,
        args.device,
        args.model_type,
        args.pretrained,
        args.model_path,
        args.dataset_type,
        args.dataset_root,
        args.batch_size,
        args.sample_size,
        args.max_target_metric_loss,
        args.metric_key,
        convert_to_bool(args.metric_increases),
        args.precision,
        args.save_dir,
    )


if __name__ == "__main__":
    main()
