import os
import json
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
import torch_directml

# =========================================================
# Leer ruta desde config.json
# =========================================================
with open("config.json", "r") as f:
    config = json.load(f)

DATASET_PATH = config["DATASET_PATH"]

# =========================================================
# Dataset personalizado para 3 clases
# =========================================================
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
                print(f"Carpeta no encontrada: {folder_path}")
                continue
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                        self.samples.append((os.path.join(root, file), label))

        if len(self.samples) == 0:
            print(f"No se encontraron imágenes en {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# =========================================================
# Transformaciones y DataLoaders
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data = ChestXrayDataset3Clases(os.path.join(DATASET_PATH, "train"), transform)
val_data   = ChestXrayDataset3Clases(os.path.join(DATASET_PATH, "val"), transform)
test_data  = ChestXrayDataset3Clases(os.path.join(DATASET_PATH, "test"), transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# =========================================================
# Dispositivo
# =========================================================
# Intentar primero CUDA (NVIDIA)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Entrenando en CUDA (NVIDIA GPU)")

else:
    try:
        import torch_directml
        device = torch_directml.device()
        print("Entrenando en DirectML (AMD/Intel GPU)")
    except ImportError:
        device = torch.device("cpu")
        print("Entrenando en CPU")

# =========================================================
# Transfer Learning: ResNet18 + capas propias
# =========================================================
class ResNet18FineTune(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet18FineTune, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Congelar todas las capas convolutivas
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Reemplazar la capa fully connected
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        # Nueva capa fully connected adaptada
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.resnet(x)
        out = self.fc(features)
        return out

model = ResNet18FineTune(num_classes=3).to(device)
for name, param in model.resnet.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

# =========================================================
# Loss con pesos para manejar desbalance de clases
# =========================================================
counts = Counter([label for _, label in train_data.samples])
weights = torch.tensor([1.0 / counts[i] if counts[i] > 0 else 0 for i in range(3)], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================================================
# Entrenamiento
# =========================================================
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
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
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100*correct/total

    # Validación
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs,1)
            val_total += labels.size(0)
            val_correct += (predicted==labels).sum().item()
    val_acc = 100*val_correct/val_total if val_total>0 else 0
    val_loss /= len(val_loader) if len(val_loader)>0 else 1

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

# =========================================================
# Evaluación final + matriz de confusión
# =========================================================
model.eval()
classes = ["NORMAL","BACTERIA","VIRUS"]
all_labels = []
all_preds = []
class_correct = defaultdict(int)
class_total = defaultdict(int)
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        for label,pred in zip(labels,predicted):
            class_total[classes[label]] += 1
            if label==pred:
                class_correct[classes[label]] += 1

accuracy = 100*correct/total
print(f"\n Accuracy final en test: {accuracy:.2f}%")
print("\n Accuracy por clase:")
for c in classes:
    acc = 100*class_correct[c]/class_total[c] if class_total[c]>0 else 0
    print(f"{c}: {acc:.2f}%")

# Matriz
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de Confusión")
plt.show()
