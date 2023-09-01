# SparseML Image Classification Integration


[SparseML] Image Classification pipeline integrates with [torch] and [torchvision] libraries to enable the sparsification of popular image classification model.
Sparsification is a powerful technique that results in faster, smaller, and cheaper deployable models. 
After training, the model can be deployed with Neural Magic's DeepSparse Engine. The engine enables inference with GPU-class performance directly on your CPU.

This integration enables spinning up one of the following end-to-end functionalities:
- **Sparsification of Popular Torchvision Models** - easily sparsify popular [torchvision] image classification models. 
- **Sparse Transfer Learning** - fine-tune a sparse backbone model (or use one of our [sparse pre-trained models](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=classification&page=1)
## Installation

We recommend using a [virtualenv] to install dependencies.
```pip install sparseml[torchvision]```

## Tutorials

- [Sparsifying PyTorch Models Using Recipes](https://github.com/neuralmagic/sparseml/blob/main/integrations/old-examples/pytorch/tutorials/sparsifying_pytorch_models_using_recipes.md)
- [Sparse Transfer Learning for Image Classification](https://github.com/neuralmagic/sparseml/blob/main/integrations/old-examples/pytorch/tutorials/classification_sparse_transfer_learning_tutorial.md)

## Getting Started
### Sparsifying Image Classification Models
In the example below, a dense [ResNet] model is trained on the [Imagenette] dataset.
By passing the recipe `zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenette/pruned-conservative?recipe_type=original` (located in [SparseZoo](https://sparsezoo.neuralmagic.com/models/cv%2Fclassification%2Fresnet_v1-50%2Fpytorch%2Fsparseml%2Fimagenette%2Fpruned-conservative))
we modify (sparsify) the training process and/or the model.
```bash
sparseml.image_classification.train \
    --recipe-path "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenette/pruned-conservative?recipe_type=original" \
    --dataset-path ./data \
    --pretrained True \
    --arch-key resnet50 \
    --dataset imagenette \
    --train-batch-size 128 \
    --test-batch-size 256 \
    --loader-num-workers 8 \
    --save-dir sparsification_example \
    --logs-dir sparsification_example \
    --model-tag resnet50-imagenette-pruned \
    --save-best-after 8         
```

### Sparse Transfer Learning

Once you sparsify a model using [SparseML], you can easily sparse fine-tune it on a new dataset.
While you are free to use your backbone, we encourage you to leverage one of our [sparse pre-trained models](https://sparsezoo.neuralmagic.com) to boost your productivity!

In the example below, we fetch a pruned [ResNet] model, pre-trained on [ImageNet] dataset. We then fine-tune the model on the [Imagenette] dataset. 
```bash
sparseml.image_classification.train \
    --recipe-path zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none?recipe_type=transfer-classification \
    --checkpoint-path zoo \
    --arch-key resnet50 \
    --model-kwargs '{"ignore_error_tensors": ["classifier.fc.weight", "classifier.fc.bias"]}' \
    --dataset imagenette \
    --dataset-path /PATH/TO/IMAGENETTE  \
    --train-batch-size 32 \
    --test-batch-size 64 \
    --loader-num-workers 0 \
    --optim Adam \
    --optim-args '{}' \
    --model-tag resnet50-imagenette-transfer-learned
```

## SparseML CLI

[SparseML] installation provides a CLI for sparsifying your models for a specific task;
 appending the `--help` argument displays a full list of options for training in [SparseML]:
```bash
sparseml.image_classification.train --help
```
output:
```bash
Usage: sparseml.image_classification.train [OPTIONS]

  PyTorch training integration with SparseML for image classification models

Options:
  --train-batch-size, --train_batch_size INTEGER
                                  Train batch size  [required]
  --test-batch-size, --test_batch_size INTEGER
                                  Test/Validation batch size  [required]
  --dataset TEXT                  The dataset to use for training, ex:
                                  `imagenet`, `imagenette`, `cifar10`, etc.
                                  Set to `imagefolder` for a generic dataset
                                  setup with imagefolder type structure like
                                  imagenet or loadable by a dataset in
                                  `sparseml.pytorch.datasets`  [required]
  --dataset-path, --dataset_path DIRECTORY
                                  The root dir path where the dataset is
                                  stored or should be downloaded to if
                                  available  [required]
  --arch_key, --arch-key TEXT     The architecture key for image
                                  classification model; example: `resnet50`,
                                  `mobilenet`. Note: Will be read from the
                                  checkpoint if not specified
  --checkpoint-path, --checkpoint_path TEXT
                                  A path to a previous checkpoint to load the
                                  state from and resume the state for. If
                                  provided, pretrained will be ignored . If
                                  using a SparseZoo recipe, can also provide
                                  'zoo' to load the base weights associated
                                  with that recipe. Additionally, can also
                                  provide a SparseZoo model stub to load model
                                  weights from SparseZoo
  ...
```

To learn about sparsification in more detail, refer to [SparseML docs](https://docs.neuralmagic.com/sparseml/)

## Once the Training is Done...

The artifacts of the training process are saved to `--save-dir` under `--model-tag`.
Once the script terminates, you should find everything required to deploy or further modify the model,
including the recipe (with the full description of the sparsification attributes), 
checkpoint files (saved in the appropriate framework format), etc.

### Exporting the Sparse Model to ONNX

The [DeepSparse] Engine uses the ONNX format to load neural networks and then 
deliver breakthrough performance for CPUs by leveraging the sparsity and quantization within a network.

The SparseML installation provides a `sparseml.image_classification.export_onnx` 
command that you can use to load the checkpoint and create a new `model.onnx` file in the same directory the
framework directory is stored. 
Be sure the `--model_path` argument points to your trained `model.pth` or `checkpoint-best.pth` file.
Both are included in `<save-dir>/<model-tag>/framework/` from the sparsification run.

```bash
sparseml.image_classification.export_onnx \
    --arch-key resnet50 \
    --dataset imagenet \
    --dataset-path ./data/imagenette-160 \
    --checkpoint-path sparsification_example/resnet50-imagenette-pruned/framework/model.pth
```

### DeepSparse Engine Deployment

Once the model is exported in the ONNX format, it is ready for deployment with the 
[DeepSparse] Engine. 

The deployment is intuitive due to the [DeepSparse] Python API.  DeepSparse can be installed via
`pip install deepsparse`.

```python
from deepsparse import Pipeline

cv_pipeline = Pipeline.create(
  task='image_classification', 
  model_path='zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none',  # Path to checkpoint or SparseZoo stub
)

input_image = "my_image.png" # path to input image 
inference = cv_pipeline(images=input_image)
```


To learn more, refer to the [appropriate documentation in the DeepSparse repository](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/image_classification/README.md).

## Support

For Neural Magic Support, sign up or log in to our [Neural Magic Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue](https://github.com/neuralmagic/sparseml/issues).


[torch]: https://pytorch.org/
[torchvision]: https://pytorch.org/vision/stable/index.html
[SparseML]: https://github.com/neuralmagic/sparseml
[SparseZoo]: https://sparsezoo.neuralmagic.com/
[ResNet]: https://arxiv.org/abs/1512.03385
[virtualenv]: https://docs.python.org/3/library/venv.html
[ImageNet]: https://www.image-net.org/
[Imagenette]: https://github.com/fastai/imagenette
[DeepSparse]: https://github.com/neuralmagic/sparseml
[DeepSparse Image Classification Documentation]: https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/image_classification/README.md
