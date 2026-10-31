import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader, random_split
from Data.class_dataset import MRIDataset
from Model.SFCN import SFCN
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Para separar el DataFrame en train, val y test



if __name__ == "__main__":

# -------- data --------
    train_df = pd.read_csv("Nico_train_val/training_nico.csv")
    val_df = pd.read_csv("Nico_train_val/val_nico.csv")


    # Extrae las listas de rutas y edades para cada set
    train_imgs = train_df["Path"].tolist()
    train_ages = train_df["Age"].tolist()
    val_imgs = val_df["Path"].tolist()
    val_ages = val_df["Age"].tolist()
    

    train_dataset = MRIDataset(train_imgs, train_ages, mode='train', enable_aug=True)
    val_dataset = MRIDataset(val_imgs, val_ages, mode='val')


    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=8
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=8
    )
    #test_loader = DataLoader(
    #    test_dataset,
    #    batch_size=8,
    #    shuffle=False,
    #    num_workers=8
    #)
# -------- model --------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(SFCN()).to(device)
    print(device)
# -------- training --------
   
    criterion = nn.SmoothL1Loss()  # Más robusta a outliers que MSE
    optimizer = torch.optim.AdamW(
        model.parameters(),  
        lr=1e-4,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6)
    model.train()
    num_epochs = 250

    # Early stopping params
    patience = 25  # Número de épocas sin mejora para detener
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
        for imgs, ages, _, _ in train_loader:
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
            for imgs, ages, _, _ in val_loader:
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
    plt.savefig("loss_plot_nico.png")  
    torch.save(model.state_dict(), 'model_nico.pth')

    # -------- Evaluación en el set de test --------
    #model.eval()
    #test_loss = 0.0
    #all_preds = []
    #all_ages = []   
    #with torch.no_grad():
    #    for imgs, ages, _, _ in test_loader:
    #        imgs = imgs.to(device)
    #        ages = ages.to(device).unsqueeze(1)
    #        preds = model(imgs)
    #        loss = criterion(preds, ages)
    #        test_loss += loss.item() * imgs.size(0)
    #        all_preds.append(preds.cpu())
    #        all_ages.append(ages.cpu())
    #test_loss /= len(test_loader.dataset)
    # Calcular MAE
    ##all_preds = torch.cat(all_preds).numpy()
    #all_ages = torch.cat(all_ages).numpy()
    #mae = mean_absolute_error(all_ages, all_preds)

    #print(f"Test Loss (MSE): {test_loss:.3f}")
    #print(f"Test MAE: {mae:.3f}")
    
