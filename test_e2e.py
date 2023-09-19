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
from sparseml.pytorch.utils import get_prunable_layers, tensor_sparsity

sml.create_session()
session = sml.active_session()

NUM_LABELS = 3
model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_LABELS)
optimizer = Adam(model.parameters(), lr=8e-3)

train_path = "/home/sadkins/.cache/huggingface/datasets/downloads/extracted/dbf92bfb2c3766fb3083a51374ad94d8a3690f53cdf0f9113a231c2351c9ff33/train"
val_path = "/home/sadkins/.cache/huggingface/datasets/downloads/extracted/510ede718de2aeaa2f9d88b0d81d88c449beeb7d074ea594bdf25a0e6a9d51d0/validation"

NUM_LABELS = 3
BATCH_SIZE = 32

# imagenet transforms
imagenet_transform = transforms.Compose([
   transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BILINEAR, max_size=None, antialias=None),
   transforms.CenterCrop(size=(224, 224)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# datasets
train_dataset = torchvision.datasets.ImageFolder(
    root=train_path,
    transform=imagenet_transform
)

val_dataset = torchvision.datasets.ImageFolder(
    root=val_path,
    transform=imagenet_transform
)

# dataloaders
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=16)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=16)

recipe = "test_e2e_recipe.yaml"
criterion = CrossEntropyLoss()



#this doubles the stages
#session.pre_initialize_structure(
#    model=model,
#    recipe=recipe,
#    framework=Framework.pytorch
#)

session_data = session.initialize(
    framework=Framework.pytorch,
    recipe=recipe,
    model=model,
    teacher_model=None,
    optimizer=optimizer,
    train_data=train_loader,
    val_data=val_loader,
    start=0.0,
    steps_per_epoch= len(train_loader) # number of times steps in called per epoch (total_data / batch_size in normal cases)
)

running_loss = 0.0
total_correct = 0
total_predictions = 0

NUM_EPOCHS = 15
device = "cuda:0"

session.state.model.model.to(device)

# loop through batches
for epoch in range(NUM_EPOCHS):
    for step, (inputs, labels) in enumerate(session.state.data.train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        session.event(event_type=EventType.BATCH_START, batch_data=(input, labels))
        session.state.optimizer.optimizer.zero_grad()

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
    print("Epoch: {} Loss: {} Accuracy: {}".format(epoch, loss, accuracy))

for (name, layer) in get_prunable_layers(session.state.model.model):
    print(f"{name}.weight: {tensor_sparsity(layer.weight).item():.4f}")