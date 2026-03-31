import sys
import os
sys.path.append(os.path.dirname(__file__))
from vit3d_pytorch import ViT3D
import torch.nn as nn

def build_vit3d():
    v = ViT3D(
        image_size = (176, 208, 176),          # image size
        patch_size = 16,   # image patch size
        num_classes = 1,   # una clase por ser regresion
        dim = 512,  #dimension del embedding
        depth = 4,  #numero de capas del transformer
        heads = 6,  #numero de cabezas de atencion
        mlp_dim = 1024, #dimension del MLP dentro del transformer
        dropout=0.1, 
        emb_dropout=0.1
    )
    return v