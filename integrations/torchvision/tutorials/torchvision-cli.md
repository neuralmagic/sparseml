# Sparsifying Image Classification Models with the CLI

This page explains how to train a sparse image classification model with SparseML's Torchvision CLI.

With SparseML, you can create a sparse model in two ways:
- **Sparse Transfer Learning**: Fine-tune a pre-sparsified model checkpoint onto a downstream dataset, while maintaining the sparsity structure of the network. 
This process works just like typical fine-tuning and is recommended for use in any scenario where there are 
[checkpoints available in SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=classification&page=1)).

- **Sparsification from Scratch**: Apply state-of-the-art training-aware pruning and quantization algorithms to arbitrary dense networks. 
This process enables you to create a sparse version of any model, but requires you to experiment with the pruning and quantization algorithms.

Let's walk through some examples using SparseML's CLI.

## Installation

Install SparseML via `pip`.

```bash
pip install sparseml[torchvision]
```

## CLI Usage

SparseML's CLI enables you to kick-off training with various utilities like creating training pipelines, dataset loading, checkpoint saving, metric reporting, and logging handled for you.

To get started, we just need to pass three required arguments: a starting checkpoint, a SparseML recipe, and a dataset.

```bash
sparseml.image_classification.train \
    --checkpoint-path [CHECKPOINT-PATH] \
    --arch-key [ARCH-KEY] \
    --recipe [RECIPE-PATH] \
    --dataset-path [DATASET-PATH]
```

The key arguments are as follows:
- `--checkpoint-path` specifies the starting model to use in the training process. It can either be a local path to a PyTorch checkpoint or a SparseZoo stub (which SparseML uses to download a PyTorch checkpoint).

- `--arch-key` specifies the torchvision architecture of the checkpoint. For example, `resnet50` or `mobilenet`.

- `--dataset-path` specifies the dataset used for training. It must be a local path to a dataset in the ImageFolder format (we will describe the format below).

- `--recipe` specifies the sparsity related parameters of the training process. It can either be a local path to a YAML recipe file or a SparseZoo stub (which SparseML uses to download a YAML recipe file). The `recipe` is the key to enabling the sparsity-related algorithms implemented by SparseML (we will explain more details on recipes below).

Run the help command to inspect the full list of arguments and configurations.
```bash
sparseml.image_classification.train --help
```

Let's dive into some concrete examples.

### Sparse Transfer Learning

#### Overview 

Sparse Transfer is quite similiar to the typical transfer learing process used to train models, where we fine-tune a pretrained checkpoint onto a smaller downstream dataset. With Sparse Transfer Learning, we simply start the fine-tuning process from a pre-sparsified checkpoint and maintain sparsity while the training process occurs.

#### Download Dataset

Let's download [FastAI's Imagenette dataset](https://github.com/fastai/imagenette). Imagenette is a subset of 10 easily classified classes from Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvf imagenette2-320.tgz
```

#### Fine-Tune The Model
We will fine-tune a 95% pruned version of ResNet-50 ([available in SparseZoo](https://sparsezoo.neuralmagic.com/models/cv%2Fclassification%2Fresnet_v1-50%2Fpytorch%2Fsparseml%2Fimagenet%2Fpruned95_quant-none)) onto Imagenette. The starting checkpoint is identified by the following SparseZoo stub:

```bash
zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none
```

The 95% pruned-quantized ResNet-50 in SparseZoo also has a sparse transfer learning recipe, which encodes parameters than instuct SparseML to maintain sparsity during the fine-tuning process. It is identified by the following SparseZoo stub:

```bash
zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification
```

Run the following to kick off training:
```bash
sparseml.image_classification.train \
    --recipe zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --checkpoint-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --arch-key resnet50 \
    --dataset-path ./imagenette2-320\
    --batch-size 32
```

SparseML downloads the starting checkpoint and transfer learning recipe from SparseZoo. It then parses the instructions in the recipe and modifies the training loop with the 
algorithms and hyperparameters specified. In the case of sparse transfer learning, the recipe instructs SparseML to maintain sparsity during
the training process and to apply quantization over the final epochs.

The script then trains for 10 epochs, converging to ~99% accuracy on the validation set.

Here's what the transfer learning recipe looks like:
```yaml
# Epoch and Learning-Rate variables
num_epochs: 10.0
init_lr: 0.0005

# quantization variables
quantization_epochs: 6.0

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0.0
    end_epoch: eval(num_epochs)

  - !LearningRateFunctionModifier
    final_lr: 0.0
    init_lr: eval(init_lr)
    lr_func: cosine
    start_epoch: 0.0
    end_epoch: eval(num_epochs)

# Phase 1 Sparse Transfer Learning / Recovery
sparse_transfer_learning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__

# Phase 2 Apply quantization
sparse_quantized_transfer_learning_modifiers:
  - !QuantizationModifier
    start_epoch: eval(num_epochs - quantization_epochs)
```

The "Modifiers" encode how SparseML should modify the training process for Sparse Transfer Learning.
- `ConstantPruningModifier` tells SparseML to pin weights at 0 over all epochs, maintaining the sparsity structure of the network
- `QuantizationModifier` tells SparseML to quanitze the weights with quantization aware training over the last 6 epochs

As a result, the final checkpoint is trained on Imagenette, with 95% of weights pruned and quantization applied!

#### Export to ONNX

To deploy with DeepSparse, we need to export the model to ONNX.

Run the following to do so:
```bash
sparseml.image_classification.export_onnx \
  --arch_key resnet50 \
  --checkpoint_path ./checkpoint.pth \
  --dataset-path ./imagenette2-320
```

#### Modify a Recipe

To transfer learn this sparsified model to other datasets you may have to adjust certain hyperparameters in this recipe and/or training script. Some considerations:

- For more complex datasets, increase the number of epochs, adjusting the learning rate step accordingly
- Adding more learning rate step milestones can lead to more jumps in accuracy
- Increase the learning rate when increasing batch size
- Increase the number of epochs if using SGD instead of the Adam optimizer
- Update the base learning rate based on the number of steps needed to train your dataset

To update a recipe, you can download the YAML file from SparseZoo, make updates to the YAML directly, and pass the local path to SparseML.

Alternatively, you can use `--recipe_args` to modify a recipe on the fly. The following runs for 15 epochs instead of 10:

```bash
sparseml.image_classification.train \
    --recipe zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --recipe_args '{"num_epochs": 15"}' \
    --checkpoint-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification \
    --arch-key resnet50 \
    --dataset-path ./imagenette2-320\
    --batch-size 32
```

The `--recipe_args` are a json parsable dictionary of recipe variable names to values to overwrite the YAML.

#### Dataset Format

SparseML's Torchvision CLI conforms to the [Torchvision ImageFolder dataset format](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html), where images are arranged into directories representing each class. To use a custom dataset with the SparseML CLI, you will need to arrange your data into this format.

For example, the following downloads Imagenette to your local directory:

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvf imagenette2-320.tgz
```

The resulting `imagenette-320` directory looks like the following (where each subdirectory like `n01440764` 
represents one of the 10 classes in the Imagenette dataset) and each `.JPEG` file is a training example.
```bash
|-train
    |-n01440764
        |-n01440764_10026.JPEG
        |-n01440764_10027.JPEG
        |...
    |-n02979186
    |-n03028079
    |-n03417042
    |-n03445777
    |-n02102040
    |-n03000684
    |-n03394916
    |-n03425413
    |-n03888257
|-val
    |-n01440764
    |-n02979186
    | ...
```

### Sparsify From Scratch with the CLI

Stay tuned for an example sparsifying from scratch with the CLI!