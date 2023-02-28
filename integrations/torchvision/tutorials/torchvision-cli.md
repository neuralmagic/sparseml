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

#### Overview 

Sparsifying a model involves removing redundant information from a 
trained model using algorithms such as pruning and quantization. The sparse models can then be deployed with DeepSparse, which implements many optimizations to increase performance via sparsity, for GPU-class performance on CPUs.

Let's try an example using SparseML's recipe to create a sparse version of ResNet-50.

#### Download ImageNet Dataset

TO BE UPDATED

#### Sparsify The Model

Recipes are SparseML's YAML-based declarative interface for specifying the sparsity-related algorithms and parameters that should be applied during the training 
process. SparseML then parses the recipes and modifies the training loop to implement the specified algorithms.

In the case of ResNet-50, SparseZoo hosts a sparsification recipe created by the Neural Magic ML team, which was used to create a 95% pruned-quantized version. It is identified by the following SparseZoo stub:

```bash
zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none
```

<details>
   <summary> Click to see the recipe</summary>

```yaml
version: 1.1.0

# General Variables
num_epochs: 206
lr_warmup_epochs: 5
init_lr: 0.0512
warmup_lr: 0.256
weight_decay: 0.00001

# Quantization variables
quantization_epochs: 6
quantization_start_epoch: eval(num_epochs - quantization_epochs)
quantization_end_epoch: eval(num_epochs)
quantization_constant_lr_epochs: 2
quantization_init_lr: 0.00005
quantization_observer_epochs: 1
quantization_keep_bn_epochs: 2

# Pruning Variables
pruning_epochs_fraction: 0.925
pruning_epochs: eval(int((num_epochs - quantization_epochs) * pruning_epochs_fraction))
pruning_start_epoch: eval(lr_warmup_epochs)
pruning_end_epoch: eval(pruning_start_epoch + pruning_epochs)
pruning_update_frequency: 5
pruning_sparsity: 0.95
pruning_final_lr: 0.0

training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)

  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: eval(lr_warmup_epochs)
    lr_func: linear
    init_lr: eval(init_lr)
    final_lr: eval(warmup_lr)

  - !LearningRateFunctionModifier
    start_epoch: eval(lr_warmup_epochs)
    end_epoch: eval(quantization_start_epoch)
    lr_func: cosine
    init_lr: eval(warmup_lr)
    final_lr: eval(pruning_final_lr)

  - !SetWeightDecayModifier
    start_epoch: 0
    end_epoch: eval(quantization_start_epoch)
    weight_decay: eval(weight_decay)

pruning_modifiers:
  - !ACDCPruningModifier
    compression_sparsity: eval(pruning_sparsity)
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    global_sparsity: True 
    update_frequency: eval(pruning_update_frequency)
    params:
      - sections.0.0.conv1.weight
      - sections.0.0.conv3.weight
      - sections.0.0.identity.conv.weight
      - sections.0.1.conv3.weight
      - sections.0.2.conv3.weight
      - sections.1.0.conv3.weight
      - sections.0.0.conv2.weight
      - sections.0.1.conv1.weight
      - sections.0.2.conv1.weight
      - sections.1.0.conv1.weight
      - sections.1.0.identity.conv.weight
      - sections.1.1.conv3.weight
      - sections.1.2.conv3.weight
      - sections.1.3.conv3.weight
      - sections.2.0.conv3.weight
      - sections.0.1.conv2.weight
      - sections.0.2.conv2.weight
      - sections.1.0.conv2.weight
      - sections.1.1.conv1.weight
      - sections.1.2.conv1.weight
      - sections.1.3.conv1.weight
      - sections.2.0.conv1.weight
      - sections.2.0.identity.conv.weight
      - sections.2.1.conv3.weight
      - sections.2.2.conv3.weight
      - sections.2.3.conv3.weight
      - sections.2.4.conv3.weight
      - sections.2.5.conv3.weight
      - sections.3.0.conv3.weight
      - sections.3.1.conv3.weight
      - sections.3.2.conv3.weight
      - sections.1.1.conv2.weight
      - sections.1.2.conv2.weight
      - sections.1.3.conv2.weight
      - sections.2.0.conv2.weight
      - sections.2.1.conv1.weight
      - sections.2.2.conv1.weight
      - sections.2.3.conv1.weight
      - sections.2.4.conv1.weight
      - sections.2.5.conv1.weight
      - sections.3.0.conv1.weight
      - sections.3.0.identity.conv.weight
      - sections.3.1.conv1.weight
      - sections.3.2.conv1.weight
      - sections.1.1.conv2.weight
      - sections.1.2.conv2.weight
      - sections.1.3.conv2.weight
      - sections.2.0.conv2.weight
      - sections.2.1.conv1.weight
      - sections.2.2.conv1.weight
      - sections.2.3.conv1.weight
      - sections.2.4.conv1.weight
      - sections.2.5.conv1.weight
      - sections.3.0.conv1.weight
      - sections.3.0.identity.conv.weight
      - sections.3.1.conv1.weight
      - sections.3.2.conv1.weight
      - sections.2.1.conv2.weight
      - sections.2.2.conv2.weight
      - sections.2.3.conv2.weight
      - sections.2.4.conv2.weight
      - sections.2.5.conv2.weight
      - sections.3.0.conv2.weight
      - sections.3.1.conv2.weight
      - sections.3.2.conv2.weight

quantization_modifiers:
  - !SetLearningRateModifier
    start_epoch: eval(quantization_start_epoch)
    learning_rate: eval(quantization_init_lr)

  - !LearningRateFunctionModifier
    final_lr: 0.0
    init_lr: eval(quantization_init_lr)
    lr_func: cosine
    start_epoch: eval(quantization_start_epoch + quantization_keep_bn_epochs)
    end_epoch: eval(num_epochs)

  - !QuantizationModifier
    start_epoch: eval(quantization_start_epoch)
    submodules:
      - input
      - sections
    disable_quantization_observer_epoch: eval(quantization_start_epoch + quantization_observer_epochs)
    freeze_bn_stats_epoch: eval(quantization_start_epoch + quantization_keep_bn_epochs)
```

Just like the Sparse Transfer Learning recipe above, the "Modifiers" encode how SparseML should modify the training process. 

In this case, however, we swapped the `!ConstantPruningModifier` for the `!ACDCPruningModifier`. Rather than telling SparseML maintaining sparsity during the training process, this modifier declares that we should apply the [AC/DC pruning algorithm](https://arxiv.org/pdf/2106.12379.pdf) to iteratively prune weights from the network at the end of each epoch until it reaches 95% sparsity.

Additionally, just like Sparsification from Scratch example, we use the `!QuantizationModifier` to apply quantization aware training (QAT) over the final epochs.

</details>

Run the following to apply the recipe:

```bash
sparseml.image_classification.train \
    --recipe zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=original \
    --arch-key resnet50 \
    --dataset-path /PATH/TO/IMAGENET  \
    --batch-size 256
```

SparseML downloads the starting checkpoint (specified by `arch-key`) and transfer learning recipe from the SparseZoo. It then parses the instructions in the recipe and modifies the training loop to apply the AC/DC and QAT algorithms based on the specifications in the recipe. As a result, the final checkpoint is trained on ImageNet, with 95% of weights pruned and quantization applied. 

This model achieves 75.8% top1 validation accuracy, recovering over 99% of the top1 validation accuracy of baseline model (76.1%).