
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader, random_split
from Data.class_dataset import MRIDataset
from Model.model import build_vit3d
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# Para separar el DataFrame en train, val y test
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

# -------- data --------
    df = pd.read_csv("Training/training_data.csv")  # Cambia la ruta al archivo CSV

    # Separar en train (70%), val (15%) y test (15%)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Extrae las listas de rutas y edades para cada set
    train_imgs = train_df["Path"].tolist()
    train_ages = train_df["Age"].tolist()
    val_imgs = val_df["Path"].tolist()
    val_ages = val_df["Age"].tolist()
    test_imgs = test_df["Path"].tolist()
    test_ages = test_df["Age"].tolist()

    train_dataset = MRIDataset(train_imgs, train_ages)
    val_dataset = MRIDataset(val_imgs, val_ages)
    test_dataset = MRIDataset(test_imgs, test_ages)

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
    test_loader = DataLoader(
        test_dataset,
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
    num_epochs = 100

    # Early stopping params
    patience = 10  # Número de épocas sin mejora para detener
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # Para plotear
    train_losses = []
    val_losses = []
    plt.ion()
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label='Train Loss')
    line2, = ax.plot([], [], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

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

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Actualizar plot
        line1.set_data(range(1, len(train_losses)+1), train_losses)
        line2.set_data(range(1, len(val_losses)+1), val_losses)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping en la época {epoch+1}. Mejor val_loss: {best_val_loss:.3f}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    plt.ioff()
    plt.show()
    torch.save(model.state_dict(), 'model.pth')

    # -------- Evaluación en el set de test --------
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for imgs, ages in test_loader:
            imgs = imgs.to(device)
            ages = ages.to(device).unsqueeze(1)
            preds = model(imgs)
            loss = criterion(preds, ages)
            test_loss += loss.item() * imgs.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.3f}")

    
