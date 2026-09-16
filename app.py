import json
import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import spaces  # Import the spaces module

# Load metadata
with open("model_info.json") as f:
    info = json.load(f)

CLASSES = info["classes"]

# Build model — move to CUDA at module level
def build_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, info["num_classes"])
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

MODEL = build_model()
MODEL = MODEL.to("cuda")  # Move to GPU at startup

# Preprocessing (must match training eval transform)
TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(info["normalize_mean"], info["normalize_std"]),
])

# Decorate the GPU function
@spaces.GPU
def classify(image):
    if image is None:
        return {}
    image = image.convert("RGB")
    with torch.no_grad():
        logits = MODEL(TF(image).unsqueeze(0).to("cuda"))
        probs = torch.softmax(logits, dim=1)[0]
    return {cls: float(p) for cls, p in zip(CLASSES, probs)}

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a car photo"),
    outputs=gr.Label(num_top_classes=5, label="Predicted Body Type"),
    title="🚗 Car Body Type Classifier",
    description=f"ResNet50 (86.6% test accuracy). Classes: {', '.join(CLASSES)}",
    allow_flagging="never",
)

demo.launch()