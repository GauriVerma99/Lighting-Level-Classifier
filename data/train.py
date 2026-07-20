import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset

# ==========================================
# Device
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# Transforms
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# Dataset
# ==========================================

base_dataset = ImageFolder("data/train")

dataset_size = len(base_dataset)

train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

torch.manual_seed(42)

indices = torch.randperm(dataset_size)

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(
    ImageFolder("data/train", transform=train_transform),
    train_indices
)

val_dataset = Subset(
    ImageFolder("data/train", transform=val_transform),
    val_indices
)

print("Training Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))

# ==========================================
# DataLoaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# ==========================================
# Model
# ==========================================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze all layers

for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4

for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classifier

model.fc = nn.Linear(model.fc.in_features, 3)

model = model.to(device)

# ==========================================
# Loss & Optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# ==========================================
# Training
# ==========================================

epochs = 15

best_accuracy = 0

for epoch in range(epochs):

    # ---------------- TRAIN ----------------

    model.train()

    running_loss = 0

    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total

    # ---------------- VALIDATION ----------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {running_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    # Save best model

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(model.state_dict(), "best_model.pth")

        print("✅ Best model saved!")

print("\n==============================")
print("Training Finished!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print("==============================")