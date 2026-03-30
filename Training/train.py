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
from sklearn.metrics import mean_absolute_error

# Para separar el DataFrame en train, val y test
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

# -------- data --------
    df = pd.read_csv("training_data.csv") 
    test_df = pd.read_csv("ext_test_data.csv")  # external testing 

    # Separar en train (90%), val (10%) 
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    #val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42) #training on full database and testing on externals

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
        num_workers=8
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8
    )

# -------- model --------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(build_vit3d()).to(device)
    print(device)
# -------- training --------
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),  
        lr=1e-4,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    model.train()
    num_epochs = 250

    # Early stopping params
    patience = 23  # Número de épocas sin mejora para detener
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
            assert not torch.isnan(imgs).any(), "Hay NaN en las imágenes"
            assert not torch.isinf(imgs).any(), "Hay Inf en las imágenes"
            assert not torch.isnan(ages).any(), "Hay NaN en las edades"
            assert not torch.isinf(ages).any(), "Hay Inf en las edades"
    
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
        scheduler.step(val_loss)

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
    plt.savefig("loss_plot.png")  
    torch.save(model.state_dict(), 'model.pth')

    # -------- Evaluación en el set de test --------
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_ages = []   
    with torch.no_grad():
        for imgs, ages in test_loader:
            imgs = imgs.to(device)
            ages = ages.to(device).unsqueeze(1)
            preds = model(imgs)
            loss = criterion(preds, ages)
            test_loss += loss.item() * imgs.size(0)
            all_preds.append(preds.cpu())
            all_ages.append(ages.cpu())
    test_loss /= len(test_loader.dataset)
    # Calcular MAE
    all_preds = torch.cat(all_preds).numpy()
    all_ages = torch.cat(all_ages).numpy()
    mae = mean_absolute_error(all_ages, all_preds)

    print(f"Test Loss (MSE): {test_loss:.3f}")
    print(f"Test MAE: {mae:.3f}")
    
