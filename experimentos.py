import os, json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Cargar configuración
with open("config.json", "r") as f:
    config = json.load(f)
data_dir = Path(config["DATASET_PATH"])

# Dataset personalizado
class ChestXrayDataset3Clases(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        labels_map = {"NORMAL": 0, "BACTERIA": 1, "VIRUS": 2}
        for folder, label in labels_map.items():
            folder_path = os.path.join(root_dir, "PNEUMONIA") if folder != "NORMAL" else os.path.join(root_dir, "NORMAL")
            if folder != "NORMAL": folder_path = os.path.join(folder_path, folder)
            if not os.path.exists(folder_path): continue
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                        self.samples.append((os.path.join(root, file), label))
        self.transform = transform

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label

# Modelo con dropout variable
class ResNet18FineTune(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        for param in self.resnet.parameters(): param.requires_grad = False
        self.resnet.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.fc(self.resnet(x))

# Transformaciones
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Cargar datasets
train_data = ChestXrayDataset3Clases(data_dir / "train", train_transform)
val_data   = ChestXrayDataset3Clases(data_dir / "val", eval_transform)
test_data  = ChestXrayDataset3Clases(data_dir / "test", eval_transform)
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en {device}")

# Experimentos
experimentos = [
    {"lr": 0.001, "dropout": 0.3, "opt": "adam"},
    {"lr": 0.001, "dropout": 0.5, "opt": "adam"},
    {"lr": 0.0005, "dropout": 0.3, "opt": "adam"},
    {"lr": 0.0005, "dropout": 0.5, "opt": "adam"},
    {"lr": 0.01, "dropout": 0.3, "opt": "sgd"},
    {"lr": 0.01, "dropout": 0.5, "opt": "sgd"},
    {"lr": 0.001, "dropout": 0.1, "opt": "adam"},
]

resultados = []

for i, config in enumerate(experimentos):
    print(f"\nExperimento {i+1}: lr={config['lr']}, dropout={config['dropout']}, opt={config['opt']}")
    model = ResNet18FineTune(num_classes=3, dropout=config["dropout"]).to(device)
    for name, param in model.resnet.named_parameters():
        if "layer4" in name: param.requires_grad = True

    counts = Counter([label for _, label in train_data.samples])
    weights = torch.tensor([1.0 / counts[i] if i in counts else 1.0 for i in range(3)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    if config["opt"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    elif config["opt"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    epochs = 3
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for images, labels in DataLoader(train_data, batch_size=32, shuffle=True):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100 * correct / total
        train_acc_list.append(train_acc)

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in DataLoader(val_data, batch_size=32, shuffle=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total
        val_acc_list.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Evaluación final en test
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in DataLoader(test_data, batch_size=32, shuffle=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    resultados.append({
        "experimento": i+1,
        "lr": config["lr"],
        "dropout": config["dropout"],
        "opt": config["opt"],
        "val_acc": val_acc_list[-1],
        "test_acc": test_acc
    })

# Mostrar resultados comparativos
print("\nResultados comparativos:")
for r in resultados:
    print(f"Exp {r['experimento']}: lr={r['lr']}, dropout={r['dropout']}, opt={r['opt']} → Val Acc: {r['val_acc']:.2f}% | Test Acc: {r['test_acc']:.2f}%")
