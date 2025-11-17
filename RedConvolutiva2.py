import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import ResNet34_Weights
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.model_selection import KFold

with open("config.json", "r") as f:
    config = json.load(f)
DATASET_PATH = config["DATASET_PATH"]

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

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
])

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


train_data = ChestXrayDataset3Clases(os.path.join(DATASET_PATH, "train"), transform=None)
val_data = ChestXrayDataset3Clases(os.path.join(DATASET_PATH, "val"), transform=None)
full_data = torch.utils.data.ConcatDataset([train_data, val_data])

test_data  = ChestXrayDataset3Clases(os.path.join(DATASET_PATH, "test"), val_transform)
test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    try:
        import torch_directml
        device = torch_directml.device()
        print("gpu")
    except ImportError:
        device = torch.device("cpu")
        print("cpu")

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
            nn.Dropout(0.45),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        features = self.resnet(x)
        return self.fc(features)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

all_train_losses, all_val_losses = [], []
all_train_accs, all_val_accs = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(full_data)):

    wandb.init(
        project="chest_xray_resnet34",
        name=f"Fold_{fold + 1}",
        config={
            "epochs": 35,
            "batch_size": 64,
            "learning_rate_fc": 1e-4,
            "learning_rate_resnet": 8e-6,
            "weight_decay": 0.0016,
            "architecture": "ResNet34",
            "k_folds": 5
        },
        reinit=True
    )

    print(f"\n--Fold {fold+1}--")

    train_subset = Subset(full_data, train_idx)
    val_subset   = Subset(full_data, val_idx)

    train_subset = TransformedSubset(train_subset, transform=train_transform)
    val_subset = TransformedSubset(val_subset, transform=val_transform)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=32, shuffle=False)

    model = ResNet34FineTune(num_classes=3).to(device)
    for name, param in model.resnet.named_parameters():
        if "layer2" in name or "layer3" in name or "layer4" in name:
            param.requires_grad = True

    labels = [full_data[i][1] for i in train_idx]
    counts = Counter(labels)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (3 * counts[i]) if counts[i] > 0 else 0 for i in range(3)],
        dtype=torch.float
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.016)
    optimizer = optim.Adam([
        {"params": model.resnet.layer2.parameters(), "lr": 8e-6},
        {"params": model.resnet.layer3.parameters(), "lr": 2e-5},
        {"params": model.resnet.layer4.parameters(), "lr": 2e-5},
        {"params": model.fc.parameters(), "lr": 1e-4}
    ], weight_decay=0.0016)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=6,
        threshold=0.002,
        min_lr=1e-6
    )

    best_val_loss = float('inf')
    patience_loss = 10
    patience_counter_loss = 0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(35):
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
        scheduler.step(val_loss)

        print(f"Fold {fold+1}, Epoch {epoch+1}: Train Loss {train_loss:.4f} Train Acc {train_acc:.2f}% "
              f"Val Loss {val_loss:.4f} Val Acc {val_acc:.2f}%")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        all_train_accs.append(train_acc)
        all_val_accs.append(val_acc)

        wandb.log({
            "epoch": epoch,
            "fold": fold + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            patience_counter_loss = 0
            torch.save(model.state_dict(), f"best_resnet34_fold{fold + 1}.pth")
        else:
            patience_counter_loss += 1
            if patience_counter_loss >= patience_loss:
                print("Early stopping activado (criterio: val_loss)")
                break

    results[fold] = best_val_loss

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='o')
    plt.title(f"Loss por época (Fold {fold+1})")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc", marker='o')
    plt.plot(val_accs, label="Val Acc", marker='o')
    plt.title(f"Accuracy por época (Fold {fold+1})")
    plt.xlabel("Época")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

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
    print(f"\nFold {fold+1} - Accuracy en test: {accuracy:.2f}%")
    print("Accuracy por clase:")
    for c in classes:
        acc = 100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0
        print(f"{c}: {acc:.2f}%")

    wandb.log({f"test_accuracy_fold{fold+1}": accuracy})

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Matriz de Confusión (Fold {fold+1})")
    plt.tight_layout()
    plt.show()


print("\nResultados por fold:")
for fold, acc in results.items():
    print(f"Fold {fold + 1}: {acc:.4f}")

mean_val_loss = sum(results.values()) / len(results)
print(f"\nMedia de validación (loss): {mean_val_loss:.4f}")
wandb.log({"mean_val_loss": mean_val_loss})


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(all_train_losses, label="Train Loss", marker='o')
plt.plot(all_val_losses, label="Val Loss", marker='o')
plt.title("Loss acumulado (todos los folds)")
plt.xlabel("Época total")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(all_train_accs, label="Train Acc", marker='o')
plt.plot(all_val_accs, label="Val Acc", marker='o')
plt.title("Accuracy acumulado (todos los folds)")
plt.xlabel("Época total")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()
