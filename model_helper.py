import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os

# -----------------------------
# Class labels
# -----------------------------
class_names = [
    'Front Breakage',
    'Front Crushed',
    'Front Normal',
    'Rear Breakage',
    'Rear Crushed',
    'Rear Normal'
]

NUM_CLASSES = len(class_names)

# -----------------------------
# Model Definition
# -----------------------------
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace FC layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Load trained model ONCE
# -----------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "saved_model_sree_varshan.pth"
)

trained_model = CarClassifierResNet()
trained_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))
)
trained_model.eval()

# -----------------------------
# Prediction function
# -----------------------------
def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = trained_model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)

    return class_names[predicted_class.item()]

