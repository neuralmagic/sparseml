
## SparseML Python API

SparseML also has a Python API, enabling you to add SparseML to your existing PyTorch training flows.

Just like the CLI, the Python API uses YAML-based recipes to encode the parameters of the sparsification process, allowing you
to add SparseML with just a few lines of code.

The `ScheduleModifierManager` class is responsible for parsing the YAML recipes and overriding the standard PyTorch model and optimizer objects, 
encoding the logic of the sparsity algorithms from the recipe. Once you have called `manager.modify`, you can then use the model and 
optimizer as usual, as SparseML abstracts away the complexity of the sparsification algorithms.

The workflow looks like this:

```python
# typical model, optimizer, dataset definition
model = Model()
optimizer = Optimizer()
train_data = TrainData()

# parse recipe + edit model/optimizer with sparsity-logic
from sparseml.pytorch.optim import ScheduledModifierManager
manager = ScheduledModifierManager.from_yaml(PATH_TO_RECIPE)
optimizer = manager.modify(model, optimizer, len(train_data))

# PyTorch training loop, using the model/optimizer as usual

# clean-up
manager.finalize(model)
```

### Sparse Transfer Learning with the Python API

Let's walk through an example running Sparse Transfer Learning with the Python API. 

The following fine-tunes a 95% pruned-quantized ResNet-50 checkpoint onto the ImageNette dataset:

```python
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sparseml.pytorch.datasets import ImagenetteDataset, ImagenetteSize
from sparseml.pytorch.models import ModelRegistry
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter
from tqdm.auto import tqdm
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
IMAGE_SIZE = 224
ZOO_STUB = "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none?recipe_type=transfer-classification"

# Step 1: Define your train and validation datasets below, here we will use Imagenette
train_dataset = ImagenetteDataset(train=True, dataset_size=ImagenetteSize.s320, image_size=IMAGE_SIZE)
val_dataset = ImagenetteDataset(train=False, dataset_size=ImagenetteSize.s320, image_size=IMAGE_SIZE)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

# Step 2: Use Model Registry to download 95% pruned+quantized ResNet-50 model
model = ModelRegistry.create(
    key="resnet50",
    pretrained_path=ZOO_STUB,
    num_classes=10,
    ignore_error_tensors=["classifier.fc.weight", "classifier.fc.bias"],
)
model.to(device) # this is just a typical PyTorch object

# Step 3: Setup Loss Function and Optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=8e-3)

# Step 4: Download Transfer Learning Recipe and Update Model/Optimizer
zoo_model = Model(ZOO_STUB)
recipe_path = zoo_model.recipes.default.path
manager = ScheduledModifierManager.from_yaml(recipe_path)
optimizer = manager.modify(model, optimizer, steps_per_epoch=len(train_loader))

# Step 5: Run Transfer Learning
def run_model_one_epoch(model, data_loader, criterion, device, train=False, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    total_correct = 0
    total_predictions = 0

    for step, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()

        outputs, _ = model(inputs)  # model returns logits and softmax as a tuple
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        total_correct += torch.sum(predictions == labels).item()
        total_predictions += inputs.size(0)
        
    loss = running_loss / (step + 1.0)
    accuracy = total_correct / total_predictions
    return loss, accuracy

epoch = manager.min_epochs
for epoch in range(manager.max_epochs):
    epoch_name = f"{epoch + 1}/{manager.max_epochs}"
    
    # run training loop
    print(f"Running Training Epoch {epoch_name}")
    train_loss, train_acc = run_model_one_epoch(model, train_loader, criterion, device, train=True, optimizer=optimizer)
    print(f"Training Epoch: {epoch_name}\nTraining Loss: {train_loss}\nTop 1 Acc: {train_acc}\n")

    # run validation loop
    print(f"Running Validation Epoch {epoch_name}")
    val_loss, val_acc = run_model_one_epoch(model, val_loader, criterion, device)
    print(f"Validation Epoch: {epoch_name}\nVal Loss: {val_loss}\nTop 1 Acc: {val_acc}\n")

manager.finalize(model)

# Step 6: Export to ONNX
save_dir = "pytorch_classification"
exporter = ModuleExporter(model, output_dir=save_dir)
exporter.export_pytorch(name="resnet50_imagenette_pruned.pth")
exporter.export_onnx(torch.randn(1, 3, 224, 224), name="model.onnx", convert_qat=True)
```

There are 6 steps:
1. Create PyTorch `DataLoaders` for ImageNette. These are familiar native PyTorch objects.

2. Use SparseML's `ModelRegistry` to download a pre-sparsified version of ResNet-50 from SparseZoo. 
The `model` object is just a standard PyTorch model with pre-trained weights loaded from SparseZoo.

3. Setup setup an Optimizer and Loss function. These are familiar native PyTorch objects.

4. Download the Transfer Learning recipe from SparseZoo (the recipe is the exact same as the one used in the CLI example above). Use 
`ScheduledModifierManager` to loads the recipe and parse the modifiers. Call `manager.modify` to update the `model` and wraps the `optimizer`
with the logic of the Sparse Transfer Learning recipe.

5. We run a typical PyTorch training loop. The wrapped `optimizer` and `model` objects are used just like usual, as SparseML has abstracted 
away the implementation of the sparsification algorithms, so the sparsity structure of the network is maintained as the fine-tuning occurs.

6. Exports the model to ONNX, it can be deployed with DeepSparse for GPU-class performance on the CPU.

You have successfully fine-tuned a 95% pruned and quantized ResNet-50 onto Imagenette!

### Sparsification from Scratch with the Python API

Let's walk through an example of Sparsifying a Model from Scratch with the Python API. 

In this case, we will show how to sparsify MobileNetV2, which does not exist in the SparseZoo. Because of the recipe-driven
approach, the workflow is very similar to that of sparse transfer learing.

#### Create a Recipe

First, since MobileNetV2 is not in the SparseZoo, we will need to create a sparsification recipe. 

The example below is the simple recipe for sparsifying model, which prunes all layers to 80% sparsity using the Gradual 
Magnitude Pruning algorithm and quantizes the weights to INT8 with Quantization Aware Training:

Save the following as `custom_recipe.yaml` in your local directory.

```yaml
# General Epoch/LR variables
num_epochs: &num_epochs 100

# Pruning variables
pruning_start_epoch: &pruning_start_epoch 0.0
pruning_end_epoch: &pruning_end_epoch 40.0
pruning_update_frequency: &pruning_update_frequency 0.4
init_sparsity: &init_sparsity 0.05
final_sparsity: &final_sparsity 0.8

training_modifiers:
  - !EpochRangeModifier
    end_epoch: *num_epochs
    start_epoch: 0.0

  - !SetLearningRateModifier
    start_epoch: 0.0
    learning_rate: 0.005
    
  - !LearningRateModifier
    start_epoch: 40.0
    lr_class: CosineAnnealingWarmRestarts
    lr_kwargs:
      lr_min: 0.00005
      cycle_epochs: 50
    init_lr: 0.005

  - !SetLearningRateModifier
    start_epoch: 90.0
    learning_rate: 0.00001
    
  - !SetLearningRateModifier
    start_epoch: 91.0
    learning_rate: 0.000001
    
  - !SetWeightDecayModifier
    start_epoch: 90.0
    weight_decay: 0.0
    
pruning_modifiers:
  - !GMPruningModifier
    params: __ALL_PRUNABLE__
    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    update_frequency: *pruning_update_frequency
    final_sparsity: *final_sparsity
    init_sparsity: *init_sparsity
        
quantization_modifiers:
  - !QuantizationModifier
    start_epoch: 90.0
```

The `GMPruningModifier` instructs SparseML to run the gradual magnitude pruning (GMP) algorithm. 
Few algorithms are better than GMP in overall results, and none beat the simplicity.
GMP works by taking a trained dense network and iteratively removes the weights closest to zero at the end of 
serveral training epochs, gradually inducing sparsity in the network. 

The `QuantizationModifier` instructs SparseML to run the quantization aware training (QAT)
algorithm. QAT works fake quantization operators are injected into the graph before quantizable 
nodes for activations, and weights are wrapped with fake quantization operators. 
The fake quantization operators interpolate the weights and activations down to INT8 on the forward pass 
but enable a full update of the weights at FP32 on the backward pass, allowing the model to adapt to the loss of 
information from quantization on the forward pass. 

In our case, the recipe instucts SparseML to run GMP over the first 40 epochs, starting with 5% sparsity and ending with 80% sparsity and
to run QAT over the last 10 epochs.

Checkout the [Recipes User Guide](/user-guide/recipes) for more details on recipes.

#### Run Training

Second, now that we have our recipe, we can run the training process just like sparse transfer learning.

The workflow looks almost identical to Sparse Transfer Learning, except we will use MobileNetV2 from Torchvision, the ImageNet
dataset, and the custom recipe.

The example below assumes the recipe from above is saved in your local directory as `custom-recipe.yaml`.

```python
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sparseml.pytorch.datasets import ImagenetteDataset, ImagenetteSize
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter
from tqdm.auto import tqdm
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
IMAGE_SIZE = 224
WEIGHTS = MobileNet_V2_Weights.IMAGENET1K_V1
RECIPE_PATH = "custom-recipe.yaml"

# Step 1: Define your train and validation datasets below, here we will use Imagenette

train_dataset = ImageNet(split="train", transform=WEIGHTS.transform)
val_dataset = ImageNet(split="val", transform=WEIGHTS.transform)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

# Step 2: Use Model Registry to download 95% pruned+quantized ResNet-50 model
model = mobilenet_v2(weights=WEIGHTS)
model.to(device) # this is just a typical PyTorch object

# Step 3: Setup Loss Function and Optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=8e-3)

# Step 4: Load Custom Recipe and Update Model/Optimizer
manager = ScheduledModifierManager.from_yaml(RECIPE)
optimizer = manager.modify(model, optimizer, steps_per_epoch=len(train_loader))

# Step 5: Run Transfer Learning
def run_model_one_epoch(model, data_loader, criterion, device, train=False, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    total_correct = 0
    total_predictions = 0

    for step, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()

        outputs, _ = model(inputs)  # model returns logits and softmax as a tuple
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        total_correct += torch.sum(predictions == labels).item()
        total_predictions += inputs.size(0)
        
    loss = running_loss / (step + 1.0)
    accuracy = total_correct / total_predictions
    return loss, accuracy

epoch = manager.min_epochs
for epoch in range(manager.max_epochs):
    epoch_name = f"{epoch + 1}/{manager.max_epochs}"
    
    # run training loop
    print(f"Running Training Epoch {epoch_name}")
    train_loss, train_acc = run_model_one_epoch(model, train_loader, criterion, device, train=True, optimizer=optimizer)
    print(f"Training Epoch: {epoch_name}\nTraining Loss: {train_loss}\nTop 1 Acc: {train_acc}\n")

    # run validation loop
    print(f"Running Validation Epoch {epoch_name}")
    val_loss, val_acc = run_model_one_epoch(model, val_loader, criterion, device)
    print(f"Validation Epoch: {epoch_name}\nVal Loss: {val_loss}\nTop 1 Acc: {val_acc}\n")

manager.finalize(model)

# Step 6: Export to ONNX
save_dir = "pytorch_classification"
exporter = ModuleExporter(model, output_dir=save_dir)
exporter.export_pytorch(name="resnet50_imagenette_pruned.pth")
exporter.export_onnx(torch.randn(1, 3, 224, 224), name="model.onnx", convert_qat=True)
```

We have the same 6 steps as before:
1. Create PyTorch `DataLoaders` for ImageNet. These are familiar native PyTorch objects.

2. Use Torchvision to download MobileNetV2 with pretrained weights.

3. Setup an Optimizer and Loss function. These are familiar native PyTorch objects.

4. Setup the `ScheduledModifierManager` with the `custom-recipe.yaml` and call `manager.modify` to update the `model` and wraps the `optimizer` 
with the logic of the Sparsification recipe (i.e. the GMP algorithm and the QAT algorithm).

5. Run a typical PyTorch training loop. The wrapped `optimizer` and `model` objects are 
used just like usual, as SparseML has abstracted away the implementation of the GMP and QAT algorithms. At the end of each epoch,
SparseML prunes the weights according to the schedule specified in the recipe.

6. Export to ONNX, enabling the model to be deployed with DeepSparse for GPU-class performance on the CPU.

You have successfully created an 80% pruned and quantized MobileNetV2!