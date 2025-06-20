import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

data_dir = 'FireDetectionDataset/images'  
batch_size = 16
img_size = 128
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    for epoch in range(num_epochs):
        total, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f} - Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "fire_model.pth")
    print(" Model saved as fire_model.pth ")


