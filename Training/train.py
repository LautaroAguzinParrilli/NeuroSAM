import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader, random_split
from Data.class_dataset import MRIDataset
from Model.model import build_vit3d
import torch.nn as nn
import pandas as pd

if __name__ == "__main__":
# -------- data --------
    df = pd.read_csv("Training/training_data.csv")  # Cambia la ruta al archivo CSV

# Extrae las listas de rutas y edades
    all_imgs = df["Path"].tolist()
    all_ages = df["Age"].tolist()
    full_dataset = MRIDataset(all_imgs, all_ages)

    # Divide en entrenamiento y validación (80% train, 20% val)
    val_split = 0.2
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

# -------- model --------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_vit3d().to(device)
# -------- training --------
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )
    model.train()
    num_epochs = 80
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for imgs, ages in train_loader:
            imgs = imgs.to(device)
            ages = ages.to(device).unsqueeze(1)
            preds = model(imgs)
            loss = criterion(preds, ages)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, ages in val_loader:
                imgs = imgs.to(device)
                ages = ages.to(device).unsqueeze(1)
                preds = model(imgs)
                loss = criterion(preds, ages)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

    torch.save(model.state_dict(), 'model.pth')

    
