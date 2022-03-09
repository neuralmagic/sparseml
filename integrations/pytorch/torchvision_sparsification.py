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
Example script for running training and optimization tasks on torchvision
classification models with an imagefolder based dataset using SparseML.
Example tasks include:
* Model pruning
* Quantization aware training
* Sparse transfer learning

More information about torchvision models can be found here:
https://pytorch.org/docs/stable/torchvision/models.html

##########
Command help:
usage: torchvision_sparseml.py [-h] --model MODEL [--recipe-path RECIPE_PATH]
                               [--image-size IMAGE_SIZE]
                               [--batch-size BATCH_SIZE]
                               [--pretrained PRETRAINED]
                               [--checkpoint-path CHECKPOINT_PATH]
                               --imagefolder-path IMAGEFOLDER_PATH
                               [--loader-num-workers LOADER_NUM_WORKERS]
                               [--loader-pin-memory LOADER_PIN_MEMORY]
                               [--model-tag MODEL_TAG] [--save-dir SAVE_DIR]

Train or finetune an image classification model from torchvision.models

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         The torchvision model class to use, ex: inception_v3,
                        resnet50, mobilenet_v2 model name is fed directly to
                        torchvision.models, more information can be found here
                        https://pytorch.org/docs/stable/torchvision/models.htm
                        l
  --recipe-path RECIPE_PATH
                        The path to the yaml file containing the sparseml
                        modifiers and schedule to apply them with
  --image-size IMAGE_SIZE
                        Size of image to use for model input. Default is 224
                        unless pytorch documentation specifies otherwise
  --batch-size BATCH_SIZE
                        Batch size to use when training model. Default is 32
  --pretrained PRETRAINED
                        Set True to use torchvisions pretrained weights, to
                        not set weights, set False. default is true.
  --checkpoint-path CHECKPOINT_PATH
                        A path to a previous checkpoint to load the state from
                        and resume the state for. If provided, pretrained will
                        be ignored
  --imagefolder-path IMAGEFOLDER_PATH
                        Path to root of dataset's generic 'image folder' path.
                        Should have an image folder structure like imagenet
                        with subdirectories 'train' and 'val' see https://pyto
                        rch.org/docs/stable/torchvision/datasets.html#imagefol
                        der
  --loader-num-workers LOADER_NUM_WORKERS
                        The number of workers to use for data loading
  --loader-pin-memory LOADER_PIN_MEMORY
                        Use pinned memory for data loading
  --model-tag MODEL_TAG
                        A tag to use for the model for saving results under
                        save-dir, defaults to the model arch and dataset used
  --save-dir SAVE_DIR   The path to the directory for saving results

##########
Example command for pruning resnet50 on an imagefolder dataset:
python integrations/pytorch/torchvision_sparsification.py \
    --recipe-path ~/sparseml_recipes/pruning_resnet50.yaml \
    --model resnet50 \
    --imagefolder-path ~/datasets/ILSVRC2012 \
    --batch-size 256
"""

import argparse
import os
import time
from types import ModuleType

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import models

from sparseml.pytorch.datasets.classification import ImageFolderDataset
from sparseml.pytorch.optim import ScheduledModifierManager, ScheduledOptimizer
from sparseml.pytorch.utils import ModuleExporter, PythonLogger, load_model
from sparseml.utils import create_dirs


MODEL_IMAGE_SIZES = {
    "inception_v3": 299,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or finetune an image classification model from "
        "torchvision.models"
    )

    # model args
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "The torchvision model class to use, ex: inception_v3, resnet50, "
            "mobilenet_v2 model name is fed directly to torchvision.models, "
            "more information can be found here "
            "https://pytorch.org/docs/stable/torchvision/models.html"
        ),
    )
    parser.add_argument(
        "--recipe-path",
        type=str,
        required=True,
        help="The path to the yaml file containing the sparseml modifiers and "
        "schedule to apply them with",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        required=False,
        default=None,
        help=(
            "Size of image to use for model input. Default is 224 unless pytorch "
            "documentation specifies otherwise"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=32,
        help="Batch size to use when training model. Default is 32",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="Set True to use torchvisions pretrained weights,"
        " to not set weights, set False. default is true.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="A path to a previous checkpoint to load the state from and "
        "resume the state for. If provided, pretrained will be ignored",
    )

    # dataset args
    parser.add_argument(
        "--imagefolder-path",
        type=str,
        required=True,
        help="Path to root of dataset's generic 'image folder' path. Should have "
        "an image folder structure like imagenet with subdirectories 'train' and 'val'"
        " see https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder",
    )
    parser.add_argument(
        "--loader-num-workers",
        type=int,
        default=4,
        help="The number of workers to use for data loading",
    )
    parser.add_argument(
        "--loader-pin-memory",
        type=bool,
        default=True,
        help="Use pinned memory for data loading",
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
        default="torchvision_sparseml_export",
        help="The path to the directory for saving results",
    )

    args = parser.parse_args()
    if args.image_size is None:
        args.image_size = (
            MODEL_IMAGE_SIZES[args.model] if args.model in MODEL_IMAGE_SIZES else 224
        )
    return args


def _get_torchvision_model(name, num_classes, pretrained=True, checkpoint_path=None):
    model_constructor = getattr(models, name, None)
    if model_constructor is None or isinstance(model_constructor, ModuleType):
        # constructor doesn't exist or is a submodule instead of function in torchvision
        raise ValueError("Torchvision model {} not found".format(name))
    # build model
    model = model_constructor(pretrained=False, num_classes=num_classes)
    if pretrained and not checkpoint_path:
        pretrained_model = model_constructor(pretrained=True, num_classes=1000)
        # fix num classes mismatch
        if num_classes == 1000:
            model = pretrained_model
        else:
            _load_matched_weights(model, pretrained_model)
        del pretrained_model

    if checkpoint_path is not None:
        load_model(checkpoint_path, model)
    return model


def _load_matched_weights(base_model, pretrained_model):
    base_dict = base_model.state_dict()
    pretrained_dict = pretrained_model.state_dict()
    for key in base_dict:
        if (
            key in pretrained_dict
            and base_dict[key].shape == pretrained_dict[key].shape
        ):
            base_dict[key] = pretrained_dict[key]
    base_model.load_state_dict(base_dict)


def _create_imagefolder_dataloader(args, train=True):
    dataset = ImageFolderDataset(
        root=args.imagefolder_path,
        train=train,
        rand_trans=train,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        pin_memory=args.loader_pin_memory,
    )

    # return dataloader, number of classes, and input image shape
    return loader, dataset.num_classes, dataset[0][0].shape


######################################################################################
# torchvision finetuning function from:                                              #
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html #
######################################################################################
def train_model(
    model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False
):
    since = time.time()

    val_acc_history = []

    # not loading best intermediate weights due to sparsity changing
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an
                    # auxiliary output. In train mode we calculate the loss by summing
                    # the final output and the auxiliary output but in testing we
                    # only consider the final output.
                    if is_inception and phase == "train":
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958  # noqa
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                # not loading best intermediate weights due to sparsity changing
                # best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    # not loading best intermediate weights due to sparsity changing
    # model.load_state_dict(best_model_wts)
    return model, val_acc_history


def _save_recipe(
    recipe_manager: ScheduledModifierManager,
    save_dir: str,
):

    recipe_save_path = os.path.join(save_dir, "recipe.yaml")
    recipe_manager.save(recipe_save_path)
    print(f"Saved recipe to {recipe_save_path}")


def main(args):
    ############################
    # logging and saving setup #
    ############################
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))

    # get unique model tag, defaults to '{model_name}'
    if not args.model_tag:
        model_tag = args.model.replace("/", ".")
        model_id = model_tag
        model_inc = 0

        while os.path.exists(os.path.join(args.save_dir, model_id)):
            model_inc += 1
            model_id = "{}__{:02d}".format(model_tag, model_inc)
    else:
        model_id = args.model_tag
    save_dir = os.path.join(save_dir, model_id)
    create_dirs(save_dir)
    print("Model id is set to {}".format(model_id))

    ###########################
    # standard training setup #
    ###########################

    # create data loaders
    train_loader, _, _ = _create_imagefolder_dataloader(args, train=True)
    val_loader, num_classes, image_shape = _create_imagefolder_dataloader(
        args, train=False
    )
    dataloaders = {"train": train_loader, "val": val_loader}

    # create model
    model = _get_torchvision_model(
        args.model,
        num_classes,
        args.pretrained,
        args.checkpoint_path,
    )
    print("created model: {}".format(model))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("using device: {}".format(device))

    # create standard SGD optimizer and cross entropy loss function
    criterion = CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(), lr=0.001, momentum=0.9
    )  # lr will be overridden by recipe

    ##########################
    # add sparseml modifiers #
    ##########################
    manager = ScheduledModifierManager.from_yaml(args.recipe_path)
    optimizer = ScheduledOptimizer(
        optimizer,
        model,
        manager,
        steps_per_epoch=len(train_loader),
        loggers=[PythonLogger()],
    )

    ########################
    # torchvision training #
    ########################
    _save_recipe(recipe_manager=manager, save_dir=save_dir)
    model, val_acc_history = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        device,
        num_epochs=manager.max_epochs,
        is_inception="inception" in args.model,
    )

    ########################
    # export trained model #
    ########################
    exporter = ModuleExporter(model, save_dir)
    sample_input = torch.randn(image_shape).unsqueeze(0)  # sample batch for ONNX export
    exporter.export_onnx(sample_input, convert_qat=True)
    exporter.export_pytorch()
    print("Model ONNX export and PyTorch weights saved to {}".format(save_dir))


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
