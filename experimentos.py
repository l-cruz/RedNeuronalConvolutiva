import itertools
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    MobileNet_V2_Weights,
    EfficientNet_B0_Weights
)


# ============================
# Dataset personalizado
# ============================
class ChestXrayDataset3Clases(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        labels_map = {"NORMAL": 0, "BACTERIA": 1, "VIRUS": 2}
        for folder, label in labels_map.items():
            folder_path = os.path.join(root_dir, "PNEUMONIA") if folder != "NORMAL" else os.path.join(root_dir, "NORMAL")
            if folder != "NORMAL":
                folder_path = os.path.join(folder_path, folder)
            if not os.path.exists(folder_path):
                continue
            for r, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                        self.samples.append((os.path.join(r, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ============================
# Transformaciones
# ============================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])

# ============================
# Dispositivo
# ============================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")

# ============================
# Modelos
# ============================
def build_model(arch, num_classes=3, dropout=0.3):
    if arch == "resnet18":
        base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
    elif arch == "resnet34":
        base = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
    elif arch == "mobilenet_v2":
        base = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(nn.Identity())
        base = nn.Sequential(base, nn.Flatten())
    elif arch == "efficientnet_b0":
        base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(nn.Identity())
        base = nn.Sequential(base, nn.Flatten())
    else:
        raise ValueError("Arquitectura desconocida")

    for p in base.parameters():
        p.requires_grad = False

    return nn.Sequential(
        base,
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )

# ============================
# Entrenamiento + evaluación
# ============================
def run_experiment(arch, opt_name, train_loader, val_loader, test_loader, device,
                   epochs=3, lr=1e-3, dropout=0.3, weight_decay=0.0):

    print(f"\n=== {arch} + {opt_name} | dropout={dropout} | wd={weight_decay} ===")
    model = build_model(arch, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()

    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Optimizador desconocido")

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validación por época
        model.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()
                _, predicted = torch.max(outputs,1)
                val_total += labels.size(0)
                val_correct += (predicted==labels).sum().item()
        val_loss = val_loss_sum / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

    # Evaluación final
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    print(f"Test Accuracy: {100*correct/total:.2f}%")

# ============================
# Main: probar todas combinaciones
# ============================
if __name__ == "__main__":
    with open("config.json","r") as f:
        config = json.load(f)
    DATASET_PATH = config["DATASET_PATH"]
    device = get_device()

    # Combinaciones anti-sobreajuste
    architectures = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
    optimizers = ["adam", "adamw", "sgd"]
    dropouts = [0.3, 0.5]
    weight_decays = [0.0, 1e-4, 5e-4]
    augmentations = [True, False]

    for arch, opt, dropout, wd, aug in itertools.product(architectures, optimizers, dropouts, weight_decays, augmentations):
        print(f"\n=== {arch} + {opt} | dropout={dropout} | wd={wd} | augment={aug} ===")

        # Transformaciones con o sin augmentación
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip() if aug else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(10) if aug else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.1, contrast=0.1) if aug else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])

        train_data = ChestXrayDataset3Clases(os.path.join(DATASET_PATH,"train"), transform)
        val_data   = ChestXrayDataset3Clases(os.path.join(DATASET_PATH,"val"), transform)
        test_data  = ChestXrayDataset3Clases(os.path.join(DATASET_PATH,"test"), transform)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

        run_experiment(arch, opt, train_loader, val_loader, test_loader, device,
                       epochs=5, lr=1e-3, dropout=dropout, weight_decay=wd)
