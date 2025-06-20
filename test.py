import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

img_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['fire', 'no_fire']  

class FireCNN(nn.Module):
    def __init__(self):
        super(FireCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (img_size // 4) * (img_size // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.model(x)

model = FireCNN().to(device)
model.load_state_dict(torch.load("fire_model.pth", map_location=device))
model.eval()
print(" Model loaded!")

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    print(f" {os.path.basename(image_path)} → {class_names[pred]} (Confidence: {confidence.item():.2f})")

test_folder = "test_images"
for img_name in os.listdir(test_folder):
    if img_name.endswith((".jpg", ".png", ".jpeg")):
        predict_image(os.path.join(test_folder, img_name))
