import sparseml.core.session as sml
from sparseml.core.framework import Framework
import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import datasets
import os
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sparseml.core.event import EventType
from sparseml.pytorch.utils import get_prunable_layers, tensor_sparsity, ModuleExporter
from statistics import mean


def get_layer_sparsities(model):
    # view sparsities
    return [ tensor_sparsity(layer.weight).item() 
        for (name, layer) in get_prunable_layers(model)
        ]
       

def match_sparsities(expected_sparsities, actual_sparsities, error=1e-5, avg=False):
    
    if avg:
        assert mean(expected_sparsities) - mean(actual_sparsities) < error, "Sparsity mismatch: {} != {}".format(mean(expected_sparsities), mean(actual_sparsities))
    else: 
        for i, (expected, actual) in enumerate(zip(expected_sparsities, actual_sparsities)):
            assert abs(expected - actual) < error, "Sparsity mismatch: {} != {} layer No: {}".format(expected, actual, i)
    
    

def save_model(model, save_dir="e2e_experiment"):
    exporter = ModuleExporter(model, output_dir=save_dir)
    exporter.export_pytorch(name="mobilenet_v2-sparse-beans.pth")
    exporter.export_onnx(torch.randn(1, 3, 224, 224), name="sparse-model.onnx")

def sparsify_model(model, sparsity=0.8):
    m = MagnitudePruningModifier(
    init_sparsity = 0.05,
    final_sparsity=sparsity,
    start_epoch= 0.0,
    end_epoch =10.0,
    update_frequency= 1.0,
    params= ["re:.*weight"],
    leave_enabled = True,
    inter_func = "cubic",
    mask_type ="unstructured",
)
    m.apply(model)
    match_sparsities([sparsity], get_layer_sparsities(model), avg=True, error=0.01)
    

NUM_LABELS = 3
BATCH_SIZE = 32
NUM_EPOCHS = 3
recipe = "./recipe.yaml"
device = "cuda:0"

# set up SparseML session
sml.create_session()
session = sml.active_session()


# download model
model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_LABELS)

from sparseml.pytorch.sparsification import MagnitudePruningModifier

sparsify_model(model=model, sparsity=0.8)
model.to(device)


# download data
beans_dataset = datasets.load_dataset("beans")
train_folder, _ = os.path.split(beans_dataset["train"][0]["image_file_path"])
train_path, _ = os.path.split(train_folder)
val_folder, _ = os.path.split(beans_dataset["validation"][0]["image_file_path"])
val_path, _ = os.path.split(train_folder)


imagenet_transform = transforms.Compose([
   transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=None),
   transforms.CenterCrop(size=(224, 224)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(
    root=train_path,
    transform=imagenet_transform
)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=16)

val_dataset = torchvision.datasets.ImageFolder(
    root=val_path,
    transform=imagenet_transform
)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=16)


# loss and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=8e-3)


# initialize session
recipe_args = {
    "warm_up_epochs": 0,
    "pruning_epochs": 3,
}
session_data = session.initialize(
    framework=Framework.pytorch,
    recipe=recipe,
    recipe_args=recipe_args,
    model=model,
    teacher_model=None,
    optimizer=optimizer,
    train_data=train_loader,
    val_data=val_loader,
    start=0.0,
    steps_per_epoch= len(train_loader)
)


# sparsity before training
expected_sparsity = get_layer_sparsities(model=model)

# loop through batches
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    total_correct = 0
    total_predictions = 0
    for step, (inputs, labels) in enumerate(session.state.data.train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        session.state.optimizer.optimizer.zero_grad()
        session.event(event_type=EventType.BATCH_START, batch_data=(inputs, labels))
        
        
        outputs = session.state.model.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        session.event(event_type=EventType.LOSS_CALCULATED, loss=loss)


        session.event(event_type=EventType.OPTIM_PRE_STEP)
        

        session.state.optimizer.optimizer.step()
        session.event(event_type=EventType.OPTIM_POST_STEP)

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        total_correct += torch.sum(predictions == labels).item()
        total_predictions += inputs.size(0)

        session.event(event_type=EventType.BATCH_END)
        
    

    loss = running_loss / (step + 1.0)
    accuracy = total_correct / total_predictions
    print("Epoch: {} Loss: {} Accuracy: {}".format(epoch + 1, loss, accuracy))
    actual_sparsity = get_layer_sparsities(model=session.state.model.model)
    match_sparsities(expected_sparsity, actual_sparsity, avg=True)
    # match_sparsities(expected_sparsity, actual_sparsity, avg=False) # will fail uncomment to see
    


# finalize session
session.finalize()
final_sparsity = get_layer_sparsities(model=session.state.model.model)
match_sparsities(expected_sparsity, final_sparsity, avg=True) # will pass
match_sparsities(expected_sparsity, final_sparsity, avg=False) # will fail
