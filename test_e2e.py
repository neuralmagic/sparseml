import sparseml.core.session as sml
from sparseml.core.framework import Framework
import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import datasets
import os
from torch.optim import Adam

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

session.initialize(
    framework=Framework.pytorch,
    recipe=recipe,
    model=model,
    teacher_model=None,
    optimizer=optimizer,
    train_data=train_loader,
    val_data=val_loader
)