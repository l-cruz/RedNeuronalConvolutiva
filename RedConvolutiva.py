import os
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
import json
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
            if folder != "NORMAL":
                folder_path = os.path.join(folder_path, folder)
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

# Transformaciones
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.GaussianBlur(kernel_size=3),
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

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en {device}")

def contar_por_clase(dataset, nombre):
    conteo = Counter([label for _, label in dataset.samples])
    clases = {0: "NORMAL", 1: "BACTERIA", 2: "VIRUS"}
    print(f"\nDistribución en {nombre}:")
    for i in range(3):
        print(f"  {clases[i]}: {conteo.get(i, 0)} imágenes")

contar_por_clase(train_data, "train")
contar_por_clase(val_data, "val")
contar_por_clase(test_data, "test")

# Modelo con fine-tuning
class ResNet18FineTune(nn.Module):
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.fc(features)

model = ResNet18FineTune(num_classes=3, dropout=0.5).to(device)

# Descongelar solo layer4
for name, param in model.resnet.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

# Pérdida ponderada por clase
counts = Counter([label for _, label in train_data.samples])
total = sum(counts.values())
weights = torch.tensor([total / counts[i] for i in range(3)], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

# Optimizador con weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Entrenamiento
epochs = 8
train_acc_list = []
val_acc_list = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
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
    train_acc_list.append(train_acc)

    # Validación
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_loader)
    val_acc_list.append(val_acc)

    print(f"Epoch [{epoch + 1}/{epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# Gráfico de accuracy
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy por época')
plt.legend()
plt.grid(True)
plt.show()

# Evaluación final en test
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
print(f"\n Accuracy final en test: {accuracy:.2f}%")
print("\n Accuracy por clase:")
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
plt.show()

torch.save(model.state_dict(), "resnet18_chestxray_best.pth")
print("\n Modelo guardado como 'resnet18_chestxray_best.pth'")
