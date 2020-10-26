"""
Script to validate a dataset's validation metrics for a given onnx model.
Uses neuralmagicML.pytorch for convenience functions to load images and metrics.
Additionally uses Neural Magic Inference Engine for inference of the model if available.


##########
Command help:
usage: detection_validation.py [-h] {neuralmagic,onnxruntime} ...

Evaluate an onnx model through Neural Magic or ONNXRuntime on a detection
dataset. Uses PyTorch datasets to load data.

positional arguments:
  {neuralmagic,onnxruntime}

optional arguments:
  -h, --help            show this help message and exit


##########
neuralmagic command help:
usage: detection_validation.py neuralmagic [-h] --onnx-file-path
                                           ONNX_FILE_PATH
                                           [--num-cores NUM_CORES] --dataset
                                           DATASET --dataset-path DATASET_PATH
                                           [--dataset-year DATASET_YEAR]
                                           [--batch-size BATCH_SIZE]
                                           [--image-size IMAGE_SIZE]
                                           [--loader-num-workers LOADER_NUM_WORKERS]
                                           [--map-iou-threshold MAP_IOU_THRESHOLD]
                                           [--map-iou-threshold-max MAP_IOU_THRESHOLD_MAX]
                                           [--results-file-path RESULTS_FILE_PATH]
                                           [--map-iou-steps MAP_IOU_STEPS]
                                           [--model-type {ssd,yolo}]

Run validation in the Neural Magic inference engine

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to run validation for
  --num-cores NUM_CORES
                        Number of cores to use the Neural Magic engine with,
                        if left unset will use all detectable cores
  --dataset DATASET     The dataset to load for validation, ex: coco, voc-
                        detection, etc. Set to imagefolder for a generic image
                        detection dataset setup
  --dataset-path DATASET_PATH
                        The root path to where the PyTorch dataset is stored
  --dataset-year DATASET_YEAR
                        the year of the dataset to use. defaults to the
                        default year
  --batch-size BATCH_SIZE
                        The batch size for the data to use to pass into the
                        model
  --image-size IMAGE_SIZE
                        The image size to use to pass into the model
  --loader-num-workers LOADER_NUM_WORKERS
                        The number of workers to use for data loading
  --map-iou-threshold MAP_IOU_THRESHOLD
                        The IoU threshold to use when calculated mean average
                        precision. To calculate mAP over a range of IoU
                        thresholds, set--map-iou-threshold-max as well.
                        Default threshold is 0.5
  --map-iou-threshold-max MAP_IOU_THRESHOLD_MAX
                        The maximum IoU to use in a range of thresholds for
                        mAP calculation
  --map-iou-steps MAP_IOU_STEPS
                        Spacing to use between steps in IoU threshold range
                        when calculating the mAP. Default is 0.05
  --results-file-path RESULTS_FILE_PATH
                        If set to a file path, will save the dictionary of
                        average precision results by IoU by class.


##########
onnxruntime command help
usage: detection_validation.py onnxruntime [-h] --onnx-file-path
                                           ONNX_FILE_PATH --dataset DATASET
                                           --dataset-path DATASET_PATH
                                           [--dataset-year DATASET_YEAR]
                                           [--batch-size BATCH_SIZE]
                                           [--image-size IMAGE_SIZE]
                                           [--loader-num-workers LOADER_NUM_WORKERS]
                                           [--map-iou-threshold MAP_IOU_THRESHOLD]
                                           [--map-iou-threshold-max MAP_IOU_THRESHOLD_MAX]
                                           [--map-iou-steps MAP_IOU_STEPS]
                                           [--results-file-path RESULTS_FILE_PATH]
                                           [--model-type {ssd,yolo}]
                                           [--no-batch-override]

Run validation in onnxruntime

optional arguments:
  -h, --help            show this help message and exit
  --onnx-file-path ONNX_FILE_PATH
                        Path to the local onnx file to run validation for
  --dataset DATASET     The dataset to load for validation, ex: coco, voc-
                        detection, etc. Set to imagefolder for a generic image
                        detection dataset setup
  --dataset-path DATASET_PATH
                        The root path to where the PyTorch dataset is stored
  --dataset-year DATASET_YEAR
                        the year of the dataset to use. defaults to the
                        default year
  --batch-size BATCH_SIZE
                        The batch size for the data to use to pass into the
                        model
  --image-size IMAGE_SIZE
                        The image size to use to pass into the model
  --loader-num-workers LOADER_NUM_WORKERS
                        The number of workers to use for data loading
  --map-iou-threshold MAP_IOU_THRESHOLD
                        The IoU threshold to use when calculated mean average
                        precision. To calculate mAP over a range of IoU
                        thresholds, set--map-iou-threshold-max as well.
                        Default threshold is 0.5
  --map-iou-threshold-max MAP_IOU_THRESHOLD_MAX
                        The maximum IoU to use in a range of thresholds for
                        mAP calculation
  --map-iou-steps MAP_IOU_STEPS
                        Spacing to use between steps in IoU threshold range
                        when calculating the mAP. Default is 0.05
  --results-file-path RESULTS_FILE_PATH
                        If set to a file path, will save the dictionary of
                        average precision results by IoU by class.
  --model-type {ssd,yolo}
                        Type of model evaluate. Options are 'yolo' and 'ssd'.
                        Default is 'ssd'
  --no-batch-override   Do not override batch dimension of the ONNX model
                        before running in ORT


##########
Example for COCO dataset with mAP@[0.5,0.95] in neuralmagic:
python detection_validation.py neuralmagic  \
    --onnx-file-path /PATH/TO/model.onnx \
    --dataset coco \
    --dataset-path /PATH/TO/coco-detection/ \
    --loader-num-workers 10 \
    --image-size 300 \
    --num-cores 8

Example for VOC dataset with mAP@0.5 in onnxruntime:
python detection_validation.py onnxruntime  \
    --onnx-file-path /PATH/TO/model.onnx \
    --dataset voc_detection \
    --dataset-path /PATH/TO/voc-detection/ \
    --loader-num-workers 10 \
    --image-size 300

Example for Yolo model and COCO dataset with mAP@0.5 in neuralmagic:
python detection_validation.py neuralmagic  \
    --onnx-file-path /PATH/TO/model.onnx \
    --model-type yolo \
    --dataset coco \
    --dataset-path /PATH/TO/coco-detection/ \
    --loader-num-workers 10 \
    --image-size 640 \
"""

import argparse
import json
from tqdm import auto

import torch
from torch.utils.data import DataLoader

from neuralmagicML import get_main_logger
from neuralmagicML.onnx.utils import ORTModelRunner, NMModelRunner
from neuralmagicML.pytorch.datasets import (
    DatasetRegistry,
    ssd_collate_fn,
    yolo_collate_fn,
)
from neuralmagicML.pytorch.utils import (
    SSDLossWrapper,
    MeanAveragePrecision,
    YoloGrids,
    postprocess_yolo,
)


LOGGER = get_main_logger()
NEURALMAGIC_COMMAND = "neuralmagic"
ONNXRUNTIME_COMMAND = "onnxruntime"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an onnx model through Neural Magic or ONNXRuntime on a "
        "detection dataset. Uses PyTorch datasets to load data."
    )

    subparsers = parser.add_subparsers(dest="command")

    neuralmagic_parser = subparsers.add_parser(
        NEURALMAGIC_COMMAND,
        description="Run validation in the Neural Magic inference engine",
    )
    onnxruntime_parser = subparsers.add_parser(
        ONNXRUNTIME_COMMAND, description="Run validation in onnxruntime",
    )

    for index, par in enumerate([neuralmagic_parser, onnxruntime_parser]):
        # model args
        par.add_argument(
            "--onnx-file-path",
            type=str,
            required=True,
            help="Path to the local onnx file to run validation for",
        )

        if index == 0:
            par.add_argument(
                "--num-cores",
                type=int,
                default=-1,
                help="Number of cores to use the Neural Magic engine with, "
                "if left unset will use all detectable cores",
            )

        # dataset args
        par.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="The dataset to load for validation, "
            "ex: coco, voc-detection, etc. "
            "Set to imagefolder for a generic image detection dataset setup",
        )
        par.add_argument(
            "--dataset-path",
            type=str,
            required=True,
            help="The root path to where the PyTorch dataset is stored",
        )
        par.add_argument(
            "--dataset-year",
            type=str,
            default=None,
            help="the year of the dataset to use. defaults to the default year",
        )
        par.add_argument(
            "--batch-size",
            type=int,
            default=16,
            help="The batch size for the data to use to pass into the model",
        )
        par.add_argument(
            "--image-size",
            type=int,
            default=300,
            help="The image size to use to pass into the model",
        )
        par.add_argument(
            "--loader-num-workers",
            type=int,
            default=4,
            help="The number of workers to use for data loading",
        )
        par.add_argument(
            "--map-iou-threshold",
            type=float,
            default=0.5,
            help="The IoU threshold to use when calculated mean average precision."
            " To calculate mAP over a range of IoU thresholds, set"
            "--map-iou-threshold-max as well. Default threshold is 0.5",
        )
        par.add_argument(
            "--map-iou-threshold-max",
            type=float,
            default=None,
            help="The maximum IoU to use in a range of thresholds for mAP calculation",
        )
        par.add_argument(
            "--map-iou-steps",
            type=float,
            default=0.05,
            help="Spacing to use between steps in IoU threshold range when calculating"
            " the mAP. Default is 0.05",
        )
        par.add_argument(
            "--results-file-path",
            type=str,
            default=None,
            help="If set to a file path, will save the dictionary of average precision"
            " results by IoU by class.",
        )
        par.add_argument(
            "--model-type",
            type=str,
            default="ssd",
            choices=["ssd", "yolo"],
            help="Type of model evaluate. Options are 'yolo' and 'ssd'. Default is 'ssd'",
        )

    onnxruntime_parser.add_argument(
        "--no-batch-override",
        action="store_true",
        help="Do not override batch dimension of the ONNX model before running in ORT",
    )

    return parser.parse_args()


def main(args):
    # dataset creation
    LOGGER.info("Creating dataset...")
    dataset_kwargs = {} if args.dataset_year is None else {"year": args.dataset_year}
    if "coco" in args.dataset.lower() or "voc" in args.dataset.lower():
        if args.model_type == "ssd":
            dataset_kwargs["preprocessing_type"] = "ssd"
        elif args.model_type == "yolo":
            dataset_kwargs["preprocessing_type"] = "yolo"
    val_dataset = DatasetRegistry.create(
        key=args.dataset,
        root=args.dataset_path,
        train=False,
        rand_trans=False,
        image_size=args.image_size,
        **dataset_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=ssd_collate_fn if args.model_type == "ssd" else yolo_collate_fn,
        shuffle=False,
        num_workers=args.loader_num_workers,
    )
    LOGGER.info("created val_dataset: {}".format(val_dataset))

    torch.multiprocessing.set_sharing_strategy("file_system")

    if args.command == NEURALMAGIC_COMMAND:
        LOGGER.info("creating model in neural magic...")
        runner = NMModelRunner(args.onnx_file_path, args.batch_size, args.num_cores)
    elif args.command == ONNXRUNTIME_COMMAND:
        LOGGER.info("creating model in onnxruntime...")
        if args.no_batch_override:
            runner = ORTModelRunner(args.onnx_file_path)
        else:
            runner = ORTModelRunner(args.onnx_file_path, batch_size=args.batch_size)
    else:
        raise ValueError("Unknown command given of {}".format(args.command))

    LOGGER.info("created runner: {}".format(runner))

    # mAP calculation setup
    if args.map_iou_threshold_max is None:
        iou_threshold = args.map_iou_threshold
    else:
        iou_threshold = [args.map_iou_threshold, args.map_iou_threshold_max]

    if args.model_type == "yolo":
        yolo_grids = YoloGrids()
        input_shape = [args.image_size, args.image_size]

        def _postprocess_yolo(outputs):
            return postprocess_yolo(outputs, input_shape, yolo_grids)

        postprocessing_fn = _postprocess_yolo
    else:

        def _postprocess_coco(outputs):
            return val_dataset.default_boxes.decode_output_batch(*outputs)

        postprocessing_fn = _postprocess_coco

    map_calculator = MeanAveragePrecision(
        postprocessing_fn, iou_threshold, args.map_iou_steps
    )
    LOGGER.info("created mAP calculator for validation: {}".format(str(map_calculator)))

    for batch, data in auto.tqdm(
        enumerate(val_loader), desc="Validation samples", total=len(val_loader)
    ):
        batch_x = {"input": data[0].numpy()}
        batch_size = data[0].shape[0]

        if batch_size != args.batch_size:
            LOGGER.warning(
                (
                    "skipping batch {} because it is not of expected batch size {}, "
                    "given {}"
                ).format(batch, args.batch_size, batch_size)
            )
            continue

        pred, pred_time = runner.batch_forward(batch_x)
        pred_pth = [torch.from_numpy(val) for val in pred.values()]
        map_calculator.batch_forward(pred_pth, data[1][-1])

    # print out results instead of log so they can't be filtered
    print("\n\ncalculating {} mAP results...".format(args.dataset))
    num_recall_levels = 101 if "coco" in args.dataset.lower() else 11
    mean_average_precision, ap_dict = map_calculator.calculate_map(num_recall_levels)

    if args.results_file_path:
        print(
            "Saving average precision by IoU threshold by class to {}".format(
                args.results_file_path
            )
        )
        with open(args.results_file_path, "w") as res_file:
            json.dump(ap_dict, res_file)
    else:
        print("average precision by IoU threshold by class:\n{}".format(ap_dict))

    print("{}: {}".format(str(map_calculator), mean_average_precision))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
