from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import shutil

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.ImageFolder(root='chest_xray/train', transform=transform)
val_data = datasets.ImageFolder(root='chest_xray/val', transform=transform)
test_data = datasets.ImageFolder(root='chest_xray/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print("Datos cargados correctamente")
print(f"Clases: {train_data.classes}")
print(f"Imágenes en train: {len(train_data)}")
print(f"Imágenes en val: {len(val_data)}")
print(f"Imágenes en test: {len(test_data)}")


def organizar_pneumonia(base_dir='chest_xray'):
    splits = ['train', 'val', 'test']

    for split in splits:
        pneu_path = os.path.join(base_dir, split, 'PNEUMONIA')
        if not os.path.exists(pneu_path):
            print(f"No se encontró la carpeta: {pneu_path}")
            continue

        virus_dir = os.path.join(pneu_path, 'VIRUS')
        bacteria_dir = os.path.join(pneu_path, 'BACTERIA')
        os.makedirs(virus_dir, exist_ok=True)
        os.makedirs(bacteria_dir, exist_ok=True)

        moved_virus, moved_bacteria, unclassified = 0, 0, 0

        for filename in os.listdir(pneu_path):
            file_path = os.path.join(pneu_path, filename)
            if os.path.isdir(file_path):
                continue

            lower_name = filename.lower()
            if 'virus' in lower_name:
                shutil.move(file_path, os.path.join(virus_dir, filename))
                moved_virus += 1
            elif 'bacteria' in lower_name:
                shutil.move(file_path, os.path.join(bacteria_dir, filename))
                moved_bacteria += 1
            else:
                print(f"Archivo no clasificado: {filename}")
                unclassified += 1

        print(f"\nReorganización completa en {pneu_path}")
        print(f"  → {moved_virus} imágenes movidas a VIRUS/")
        print(f"  → {moved_bacteria} imágenes movidas a BACTERIA/")
        if unclassified > 0:
            print(f"{unclassified} archivos sin clasificar")



organizar_pneumonia('chest_xray')
