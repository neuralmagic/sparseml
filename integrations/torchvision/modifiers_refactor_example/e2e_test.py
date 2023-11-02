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


def main():
    import os

    import datasets
    import torch
    import torchvision
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torchvision import transforms

    import sparseml.core.session as session_manager
    from sparseml.core.event import EventType
    from sparseml.core.framework import Framework
    from sparseml.pytorch.utils import (
        ModuleExporter,
        get_prunable_layers,
        tensor_sparsity,
    )

    NUM_LABELS = 3
    BATCH_SIZE = 32
    NUM_EPOCHS = 12
    recipe = "e2e_recipe.yaml"
    device = "cuda:0"

    # set up SparseML session
    session_manager.create_session()
    session = session_manager.active_session()

    # download model
    model = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.DEFAULT
    )
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_LABELS)
    model.to(device)

    # download data
    beans_dataset = datasets.load_dataset("beans")
    train_folder, _ = os.path.split(beans_dataset["train"][0]["image_file_path"])
    train_path, _ = os.path.split(train_folder)
    val_folder, _ = os.path.split(beans_dataset["validation"][0]["image_file_path"])
    val_path, _ = os.path.split(train_folder)

    # dataloaders
    imagenet_transform = transforms.Compose(
        [
            transforms.Resize(
                size=256,
                interpolation=transforms.InterpolationMode.BILINEAR,
                max_size=None,
                antialias=None,
            ),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        root=train_path, transform=imagenet_transform
    )
    train_loader = DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=16
    )

    val_dataset = torchvision.datasets.ImageFolder(
        root=val_path, transform=imagenet_transform
    )
    val_loader = DataLoader(
        val_dataset, BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=16
    )

    # loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=8e-3)

    # initialize session
    recipe_args = {"warm_up_epochs": 5, "start_quant_epoch": 3, "pruning_epochs": 5}
    _ = session.initialize(
        framework=Framework.pytorch,
        recipe=recipe,
        recipe_args=recipe_args,
        model=model,
        teacher_model=None,
        optimizer=optimizer,
        train_data=train_loader,
        val_data=val_loader,
        start=0.0,
        steps_per_epoch=len(train_loader),
    )

    # loop through batches
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        total_correct = 0
        total_predictions = 0
        for step, (inputs, labels) in enumerate(session.state.data.train):
            inputs = inputs.to(device)
            labels = labels.to(device)
            session.state.optimizer.optimizer.zero_grad()
            session.event(event_type=EventType.BATCH_START, batch_data=(input, labels))

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

    # finalize session
    session.finalize()

    # view sparsities
    for (name, layer) in get_prunable_layers(session.state.model.model):
        print(f"{name}.weight: {tensor_sparsity(layer.weight).item():.4f}")

    # save sparsified model
    save_dir = "e2e_experiment"
    exporter = ModuleExporter(model, output_dir=save_dir)
    exporter.export_pytorch(name="mobilenet_v2-sparse-beans.pth")
    exporter.export_onnx(torch.randn(1, 3, 224, 224), name="sparse-model.onnx")


if __name__ == "__main__":
    main()
