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
Utility script to export a model to onnx and also store sample inputs/outputs
"""

from tqdm import tqdm

from helpers import *


CURRENT_TASK = "export"


def parse_args():
    """
    Utility function to add and parse export specific command-line args
    """
    parser = argparse.ArgumentParser(__doc__)

    add_universal_args(parser=parser, task=CURRENT_TASK)
    add_export_specific_args(parser=parser)
    args = parser.parse_args()
    args = parse_ddp_args(args, task=CURRENT_TASK)

    append_preprocessing_args(args)

    return args


def export(args, model, val_loader, save_dir):
    exporter = ModuleExporter(model, save_dir)

    # export PyTorch state dict
    LOGGER.info("exporting pytorch in {}".format(save_dir))
    exporter.export_pytorch(
        use_zipfile_serialization_if_available=(
            args.use_zipfile_serialization_if_available
        )
    )
    onnx_exported = False

    for batch, data in tqdm(
        enumerate(val_loader),
        desc="Exporting samples",
        total=args.num_samples if args.num_samples > 1 else 1,
    ):
        if not onnx_exported:
            # export onnx file using first sample for graph freezing
            LOGGER.info("exporting onnx in {}".format(save_dir))
            exporter.export_onnx(data[0], opset=args.onnx_opset, convert_qat=True)
            onnx_exported = True

        if args.num_samples > 0:
            exporter.export_samples(
                sample_batches=[data[0]], sample_labels=[data[1]], exp_counter=batch
            )


def main():
    args_ = parse_args()
    distributed_setup(args_.local_rank)
    model, save_dir, val_loader = export_setup(args_)
    export(args_, model, val_loader, save_dir)


def export_setup(args_):
    save_dir, loggers = get_save_dir_and_loggers(args_, task=CURRENT_TASK)
    input_shape = ModelRegistry.input_shape(args_.arch_key)
    image_size = input_shape[1]  # assume shape [C, S, S] where S is the image size
    (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
    ) = get_train_and_validation_loaders(args_, image_size, task=CURRENT_TASK)
    # model creation
    num_classes = infer_num_classes(args_, train_dataset, val_dataset)
    model = create_model(args_, num_classes)
    return model, save_dir, val_loader


if __name__ == "__main__":
    main()
