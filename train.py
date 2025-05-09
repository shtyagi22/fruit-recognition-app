# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

train_dir = "fruits/Training"
dataset = datasets.ImageFolder(train_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training...")
for epoch in range(5):  # Keep small for now
    total, correct = 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}: Accuracy = {100 * correct / total:.2f}%")

# Save the model and class names
torch.save(model.state_dict(), "model.pth")
with open("classes.txt", "w") as f:
    f.write("\n".join(dataset.classes))
