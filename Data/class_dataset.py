import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import nibabel as nib
import numpy as np
 
 
def apply_fda(src_img, trg_img, beta=0.05):
    """
    Intercambia las bajas frecuencias del espectro de amplitud entre dos imágenes (FDA).
    src_img: imagen a la que se le transfiere el contraste/estilo.
    trg_img: imagen donante de amplitud.
    Acepta tensores (C, H, W) o (C, D, H, W): la FFT se aplica sobre las dos últimas dimensiones.
    """
    fft_src = torch.fft.fftn(src_img, dim=(-2, -1))
    fft_trg = torch.fft.fftn(trg_img, dim=(-2, -1))
 
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg = torch.abs(fft_trg)
 
    h, w = src_img.shape[-2:]
    b = max(1, int(np.floor(min(h, w) * beta)))
 
    amp_src = amp_src.clone()
    amp_src[..., :b, :b] = amp_trg[..., :b, :b]
    amp_src[..., :b, -b:] = amp_trg[..., :b, -b:]
    amp_src[..., -b:, :b] = amp_trg[..., -b:, :b]
    amp_src[..., -b:, -b:] = amp_trg[..., -b:, -b:]
 
    fft_src_aug = amp_src * torch.exp(1j * pha_src)
    return torch.fft.ifftn(fft_src_aug, dim=(-2, -1)).real
 
 
class MRIDataset(Dataset):
    def __init__(self, img_paths, ages, target_shape=(176, 208, 176),
                 mode='val', enable_aug=True):
        """
        mode: 'train' | 'val' | 'test'. La augmentation solo se aplica en 'train'.
        enable_aug: permite desactivar la augmentation incluso en modo 'train'.
        """
        self.img_paths = img_paths
        self.ages = ages
        self.target_shape = target_shape
        self.mode = mode
        self.enable_aug = enable_aug
 
    def __len__(self):
        return len(self.img_paths)
 
    def crop_center(self, img, target_shape):
        d, h, w = img.shape
        td, th, tw = target_shape
        start_d = (d - td) // 2
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        return img[start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]
 
    def _load_volume(self, img_path):
        """Carga y recorta un volumen. Devuelve (tensor (1,D,H,W), nii)."""
        nii = nib.load(img_path)
        img = nii.get_fdata().astype(np.float32)
 
        # Crop central
        img = self.crop_center(img, self.target_shape)
 
        # (C, D, H, W)
        return torch.tensor(img).unsqueeze(0), nii
 
    def _augment(self, img):
        # 1. Ruido gaussiano aditivo
        if torch.rand(1) < 0.5:
            img = img + (torch.randn_like(img) * 0.01)
 
        # 2. Desenfoque gaussiano (2D, aplicado corte por corte sobre H, W)
        if torch.rand(1) < 0.3:
            kernel_size = int(np.random.choice([3, 5]))
            sigma = float(torch.empty(1).uniform_(0.1, 1.1))
            img = TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size],
                                   sigma=[sigma, sigma])
 
        # 3. Ajuste de nitidez
        # adjust_sharpness exige (..., 1 o 3, H, W): se trata cada corte D como un
        # "batch" de imágenes de un canal, es decir (D, 1, H, W).
        if torch.rand(1) < 0.3:
            sharp_factor = float(torch.empty(1).uniform_(1.5, 3.0))
            img = torch.cat([
                TF.adjust_sharpness(img[c].unsqueeze(1), sharp_factor).squeeze(1).unsqueeze(0)
                for c in range(img.shape[0])
            ], dim=0)
 
        # 4. Fourier Domain Adaptation (FDA)
        if torch.rand(1) < 0.8:
            try:
                random_idx = np.random.randint(0, len(self.img_paths))
                trg_img, _ = self._load_volume(self.img_paths[random_idx])
                img = apply_fda(img, trg_img, beta=0.05)
            except Exception:
                pass
 
        return img
 
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img, nii = self._load_volume(img_path)
 
        if self.mode == 'train' and self.enable_aug:
            img = self._augment(img)
 
        age = torch.tensor(self.ages[idx]).float()
        affine = torch.tensor(nii.affine).float()
 
        return img, age, affine, img_path


