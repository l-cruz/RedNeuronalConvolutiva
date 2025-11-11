import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import ResNet34_Weights
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Leer ruta desde config.json
with open("config.json", "r") as f:
    config = json.load(f)
DATASET_PATH = config["DATASET_PATH"]

# Dataset personalizado
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
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                        self.samples.append((os.path.join(root, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformaciones
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# Cargar datasets
train_data = ChestXrayDataset3Clases(os.path.join(DATASET_PATH, "train"), train_transform)
val_data   = ChestXrayDataset3Clases(os.path.join(DATASET_PATH, "val"), val_transform)
test_data  = ChestXrayDataset3Clases(os.path.join(DATASET_PATH, "test"), val_transform)

# Mostrar cantidad de imágenes por clase
counts = Counter([label for _, label in train_data.samples])
labels_map_inv = {0: "NORMAL", 1: "BACTERIA", 2: "VIRUS"}
print("\nCantidad de imágenes por clase en el conjunto de entrenamiento:")
for i in range(3):
    print(f"{labels_map_inv[i]}: {counts[i]}")

# Sampler balanceado
class_weights = [1.0 / counts[i] for i in range(3)]
sample_weights = [class_weights[label] for _, label in train_data.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=64, sampler=sampler)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

# Dispositivo
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cpu")

# Modelo con ResNet34
class ResNet34FineTune(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet34FineTune, self).__init__()
        self.resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad = False
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        features = self.resnet(x)
        return self.fc(features)

model = ResNet34FineTune(num_classes=3).to(device)
for name, param in model.resnet.named_parameters():
    if "layer2" in name or "layer3" in name or "layer4" in name:
        param.requires_grad = True

# Loss con pesos
total = sum(counts.values())
weights = torch.tensor(
    [total / (3 * counts[i]) if counts[i] > 0 else 0 for i in range(3)],
    dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
optimizer = optim.Adam([
    {"params": model.resnet.layer2.parameters(), "lr": 1e-5},
    {"params": model.resnet.layer3.parameters(), "lr": 1e-5},
    {"params": model.resnet.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 2e-4}
], weight_decay=0.005)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18)

# Entrenamiento
epochs = 18
best_val_loss = float("inf")
patience_counter = 0
train_accs, val_accs = [], []
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
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
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validación
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_resnet34_model.pth")  # Guardar mejor modelo
    else:
        patience_counter += 1
        if patience_counter >= 5:
            print("Early stopping activado")
            break

# Evaluación final
model.eval()
classes = ["NORMAL", "BACTERIA", "VIRUS"]
all_labels, all_preds = [], []
class_correct = defaultdict(int)
class_total = defaultdict(int)
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        for label, pred in zip(labels, predicted):
            class_total[classes[label]] += 1
            if label == pred:
                class_correct[classes[label]] += 1

accuracy = 100 * correct / total
print(f"\nAccuracy final en test: {accuracy:.2f}%")
print("\nAccuracy por clase:")
for c in classes:
    acc = 100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0
    print(f"{c}: {acc:.2f}%")

# Matriz de confusión
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()

# Gráficas de entrenamiento
plt.figure(figsize=(10, 4))

# Loss por época
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Val Loss", marker='o')
plt.title("Loss por época")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()

# Accuracy por época
plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc", marker='o')
plt.plot(val_accs, label="Val Acc", marker='o')
plt.title("Accuracy por época")
plt.xlabel("Época")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()
