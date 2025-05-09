import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

import os

# For HF Spaces: avoid errors if run in limited environments
if not os.path.exists("model.pth") or not os.path.exists("classes.txt"):
    raise FileNotFoundError("Please upload model.pth and classes.txt to your Space.")


# ✅ Preprocessing: match validation pipeline
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ✅ Load class labels
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()
print(f"✅ Loaded {len(classes)} class labels.")

# ✅ Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()
print("✅ Model loaded and ready.")

# ✅ Inference function
def classify_image(img):
    print("🖼️ Received image for classification.")
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        preds = model(img)
        predicted_idx = torch.argmax(preds, 1).item()
    prediction = classes[predicted_idx]
    print(f"🎯 Predicted: {prediction}")
    return prediction

# ✅ Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs="image",
    outputs="text",
    title="🍎 Fruit Classifier",
    description="Upload a fruit image and let the model tell you what it is!"
)

print("🚀 Launching web app...")
interface.launch()
