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
import wandb

#Leer ruta desde config.json
with open("config.json", "r") as f:
    config = json.load(f)

DATASET_PATH = config["DATASET_PATH"]

#Inicializar W&B
wandb.init(
    project="clasificacion-radiografias",
    name="resnet18_finetune_3clases",
    config={
        "epochs": 40,
        "batch_size": 64,
        "lr": 0.0001,
        "optimizer": "SGD",
        "architecture": "ResNet18",
        "dataset": "ChestXrayDataset3Clases",
        "patience": 8
    }
)
config_wb = wandb.config

#Dataset personalizado
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


# Transformaciones
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

train_data = ChestXrayDataset3Clases(os.path.join(DATASET_PATH, "train"), train_transform)
val_data   = ChestXrayDataset3Clases(os.path.join(DATASET_PATH, "val"), val_transform)
test_data  = ChestXrayDataset3Clases(os.path.join(DATASET_PATH, "test"), val_transform)

train_loader = DataLoader(train_data, batch_size=config_wb.batch_size, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

#Dispositivo
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

#Modelo
class ResNet18FineTune(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet18FineTune, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad = False
        in_features = self.resnet.fc.in_features
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
        out = self.fc(features)
        return out

model = ResNet18FineTune(num_classes=3).to(device)
for name, param in model.resnet.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

#Pesos y optimizador
counts = Counter([label for _, label in train_data.samples])
total = sum(counts.values())
num_classes = 3
weights = torch.tensor(
    [total / (num_classes * counts[i]) for i in range(num_classes)],
    dtype=torch.float
).to(device)
weights = weights / weights.mean()
weights[0] *= 1.5

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.SGD([
    {'params': model.resnet.layer3.parameters(), 'lr': config_wb.lr / 4},
    {'params': model.resnet.layer4.parameters(), 'lr': config_wb.lr / 2},
    {'params': model.fc.parameters(), 'lr': config_wb.lr * 2}
], momentum=0.9, weight_decay=0.001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,
    T_mult=2
)

best_val_loss = float('inf')
epochs_no_improve = 0
patience = config_wb.patience

#Entrenamiento
epochs = config_wb.epochs
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
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    #Validación
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
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total if val_total > 0 else 0
    val_loss /= len(val_loader) if len(val_loader) > 0 else 1

    #Registrar métricas en W&B
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc
    })

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\n Early stopping activado: no hay mejora en {patience} épocas consecutivas.")
            break

#Evaluación final
model.eval()
classes = ["NORMAL", "BACTERIA", "VIRUS"]
all_labels, all_preds = [], []
class_correct, class_total = defaultdict(int), defaultdict(int)
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        for label,pred in zip(labels, predicted):
            class_total[classes[label]] += 1
            if label == pred:
                class_correct[classes[label]] += 1

accuracy = 100 * correct / total
wandb.log({"test_accuracy": accuracy})
print(f"\n Accuracy final en test: {accuracy:.2f}%")

print("\n Accuracy por clase:")
for c in classes:
    acc = 100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0
    print(f"{c}: {acc:.2f}%")
    wandb.log({f"accuracy_{c}": acc})

#Matriz de confusión
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Matriz de Confusión")
plt.tight_layout()
wandb.log({"matriz_confusion": wandb.Image(plt)})
plt.show()
wandb.finish()