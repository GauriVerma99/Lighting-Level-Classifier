import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Read test CSV
test_df = pd.read_csv("data/test.csv")

predictions = []

with torch.no_grad():

    for img_id in test_df["id"]:

        img_path = f"data/test/{img_id}.png"

        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        output = model(image)

        pred = torch.argmax(output, dim=1).item()

        predictions.append(pred)

# Create submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "label": predictions
})

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")
