# app.py
import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 131)  # Placeholder
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Load class labels
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])


def classify_image(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)  # Convert NumPy array to PIL Image
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        preds = model(img)
        predicted = torch.argmax(preds, 1).item()
    return classes[predicted]


interface = gr.Interface(fn=classify_image, inputs="image", outputs="text", title="Fruit Classifier")
interface.launch(share=True)
