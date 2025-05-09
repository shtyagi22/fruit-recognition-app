import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from pathlib import Path

# ✅ List of allowed fruit class folder names
allowed_classes = [
    "Apple Golden 1", "Apple Golden 2", "Apple Red 1", "Apple Red 2", "Apple Red 3",
    "Apricot", "Avocado", "Banana", "Blueberry", "Cactus fruit", "Cherry 1", "Cherry 2",
    "Clementine", "Dates", "Fig", "Grape Pink", "Grape White", "Guava", "Kiwi",
    "Lemon", "Limes", "Lychee", "Mandarine", "Mango", "Melon Piel de Sapo",
    "Nectarine", "Orange", "Papaya", "Passion Fruit", "Peach", "Pear", "Pineapple",
    "Plum", "Pomegranate", "Quince", "Raspberry", "Strawberry", "Tamarillo", "Tomato 1",
    "Tomato 2", "Tomato 3", "Tomato 4", "Watermelon"
]

# ✅ Augmented training transform
train_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ✅ Custom filtered dataset class
class FilteredDataset(datasets.ImageFolder):
    def find_classes(self, directory):
        # Override class list to only include allowed fruits
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and d.name in allowed_classes]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

print("📂 Loading filtered fruit-only dataset...")
data_dir = "fruits/Training"
dataset = FilteredDataset(data_dir, transform=train_transform)
print(f"✅ Loaded {len(dataset)} images across {len(dataset.classes)} fruit classes.")

# ✅ Train/Val split
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Train model
epochs = 5
print(f"🚀 Training on fruits-only dataset for {epochs} epochs...\n")
for epoch in range(epochs):
    model.train()
    train_correct, train_total = 0, 0
    print(f"🔄 Epoch {epoch+1}/{epochs}")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"   Batch {batch_idx+1}/{len(train_loader)}")

    # ✅ Validate
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    print(f"✅ Epoch {epoch+1} complete | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%\n")

# ✅ Save model and fruit-only classes
torch.save(model.state_dict(), "model.pth")
with open("classes.txt", "w") as f:
    f.write("\n".join(dataset.classes))

print("💾 Model and fruit class list saved as model.pth and classes.txt")
