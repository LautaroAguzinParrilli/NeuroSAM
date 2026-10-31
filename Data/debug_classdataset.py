import os
import random
 
import matplotlib
matplotlib.use("Agg")  # sin ventana, solo guarda archivos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
 
from class_dataset import MRIDataset
 
# -------- configuración --------
CSV_PATH = "/data/lautaro/NeuroSAM/Training/Nico_train_val/training_nico.csv"   # debe tener las columnas "Path" y "Age"
N_SAMPLES = 100
OUTPUT_DIR = "imagenes"
SEED = 42                        # poné None para una muestra distinta en cada ejecución
 
 
def nombre_base(path):
    nombre = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if nombre.endswith(ext):
            return nombre[: -len(ext)]
    return os.path.splitext(nombre)[0]
 
 
def guardar_cortes(vol, titulo, out_path):
    """
    vol: array 3D (D, H, W). Guarda una PNG con los cortes centrales
    axial, coronal y sagital.
    Se asume que los ejes del volumen son (x, y, z) como en nibabel
    (orientación tipo RAS, la del espacio MNI tras el preprocesamiento).
    """
    x, y, z = [s // 2 for s in vol.shape]
 
    cortes = {
        "Axial": vol[:, :, z].T,
        "Coronal": vol[:, y, :].T,
        "Sagital": vol[x, :, :].T,
    }
 
    # Rango de visualización común a los tres cortes (robusto a outliers)
    vmin, vmax = np.percentile(vol, [1, 99])
 
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, (nombre, corte) in zip(axes, cortes.items()):
        ax.imshow(corte, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(nombre)
        ax.axis("off")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
 
 
if __name__ == "__main__":
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    df = pd.read_csv(CSV_PATH)
    n = min(N_SAMPLES, len(df))
    muestra = df.sample(n=n, random_state=SEED)
 
    img_paths = muestra["Path"].tolist()
    ages = muestra["Age"].tolist()
 
    # mode='train' y enable_aug=True para ver el resultado de la augmentation.
    # Para ver las imágenes sin aumentar, usá mode='val'.
    dataset = MRIDataset(img_paths, ages, mode="train", enable_aug=True)
 
    for i in range(len(dataset)):
        img, age, affine, path = dataset[i]   # img: (1, D, H, W)
        vol = img[0].numpy()
 
        titulo = f"{nombre_base(path)} | edad: {age.item():.1f}"
        out_path = os.path.join(OUTPUT_DIR, f"{i:03d}_{nombre_base(path)}.png")
        guardar_cortes(vol, titulo, out_path)
 
        print(f"[{i + 1}/{len(dataset)}] guardada {out_path}")
 
    print(f"Listo. {len(dataset)} imágenes guardadas en '{OUTPUT_DIR}/'")