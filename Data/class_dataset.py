import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import nibabel as nib
import numpy as np


# -------- Parámetros de la augmentation (suaves) --------
AUG_PARAMS = {
    "noise_p": 0.5,          # probabilidad del ruido gaussiano
    "noise_std": 0.01,      # desvío del ruido, relativo al desvío de la imagen
    "blur_p": 0.3,           # probabilidad del desenfoque gaussiano
    "blur_kernels": [3],     # tamaños de kernel posibles
    "blur_sigma": (0.1, 1.1),
    "sharp_p": 0.3,          # probabilidad del ajuste de nitidez
    "sharp_factor": (1.5, 3.0),
    "fda_p": 0.8,            # probabilidad de FDA
    "fda_beta": 0.05,        # tamaño de la ventana de bajas frecuencias
}


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


def unsharp_mask(img, factor, kernel_size=3, sigma=1.0):
    """
    Ajuste de nitidez sin recorte de valores (funciona con cualquier rango de intensidades).
    factor = 1.0 no modifica la imagen; factor > 1.0 realza los bordes.
    Opera en 2D sobre las dos últimas dimensiones (H, W).
    """
    blurred = TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size],
                               sigma=[sigma, sigma])
    return img + (factor - 1.0) * (img - blurred)


class MRIDataset(Dataset):
    def __init__(self, img_paths, ages, target_shape=(176, 208, 176),
                 mode='val', enable_aug=True, aug_params=None):
        """
        mode: 'train' | 'val' | 'test'. La augmentation solo se aplica en 'train'.
        enable_aug: permite desactivar la augmentation incluso en modo 'train'.
        aug_params: dict para sobreescribir parámetros de AUG_PARAMS.
        """
        self.img_paths = img_paths
        self.ages = ages
        self.target_shape = target_shape
        self.mode = mode
        self.enable_aug = enable_aug
        self.aug = {**AUG_PARAMS, **(aug_params or {})}

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
        """Carga, recorta y normaliza (z-score) un volumen. Devuelve (tensor (1,D,H,W), nii)."""
        nii = nib.load(img_path)
        img = nii.get_fdata().astype(np.float32)

        # Crop central
        img = self.crop_center(img, self.target_shape)

        # (C, D, H, W), con z-score: así la imagen y la donante de FDA
        # quedan en la misma escala de intensidades
        return self._zscore(torch.tensor(img).unsqueeze(0)), nii

    @staticmethod
    def _zscore(img):
        """Normalización z-score sobre todo el volumen."""
        return (img - img.mean()) / (img.std() + 1e-8)

    def _augment(self, img):
        p = self.aug

        # 1. Ruido gaussiano aditivo (relativo al desvío de la imagen)
        if torch.rand(1) < p["noise_p"]:
            img = img + torch.randn_like(img) * (p["noise_std"] * img.std())

        # 2. Desenfoque gaussiano (2D, aplicado corte por corte sobre H, W)
        if torch.rand(1) < p["blur_p"]:
            kernel_size = int(np.random.choice(p["blur_kernels"]))
            sigma = float(torch.empty(1).uniform_(*p["blur_sigma"]))
            img = TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size],
                                   sigma=[sigma, sigma])

        # 3. Ajuste de nitidez (unsharp mask propio, sin recorte)
        if torch.rand(1) < p["sharp_p"]:
            sharp_factor = float(torch.empty(1).uniform_(*p["sharp_factor"]))
            img = unsharp_mask(img, sharp_factor)

        # 4. Fourier Domain Adaptation (FDA)
        if torch.rand(1) < p["fda_p"]:
            try:
                random_idx = np.random.randint(0, len(self.img_paths))
                trg_img, _ = self._load_volume(self.img_paths[random_idx])
                img = apply_fda(img, trg_img, beta=p["fda_beta"])
            except Exception:
                pass

        return img

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img, nii = self._load_volume(img_path)

        if self.mode == 'train' and self.enable_aug:
            img = self._augment(img)

        # Z-score final fuera de la augmentation: se aplica a TODAS las imágenes
        # (train/val/test) para que el modelo reciba media 0 y desvío 1 exactos,
        # ya que la augmentation altera levemente esas estadísticas
        img = self._zscore(img)

        age = torch.tensor(self.ages[idx]).float()
        affine = torch.tensor(nii.affine).float()

        return img, age, affine, img_path