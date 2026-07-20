import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Classes
classes = ['bright', 'dark', 'normal']

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Check input
if len(sys.argv) != 2:
    print("Usage:")
    print("python predict.py image_path")
    exit()

image_path = sys.argv[1]

# Load image
image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

print(f"Prediction : {classes[predicted.item()]}")
print(f"Confidence : {confidence.item()*100:.2f}%")