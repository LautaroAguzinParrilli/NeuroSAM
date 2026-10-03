import os
import sys
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img

from torch.utils.data import DataLoader

from matplotlib.colors import TwoSlopeNorm

# Zennit
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlus

# ---------------------------------------------------------------------
# IMPORTAR TU MODELO Y DATASET
# ---------------------------------------------------------------------

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Model.SFCN import SFCN
from Data.class_dataset import MRIDataset


# ---------------------------------------------------------------------
# VISUALIZACIÓN
# ---------------------------------------------------------------------

def save_relevance_png_3views(rel_data, out_path, perc_clip=99):

    vmax = np.percentile(np.abs(rel_data), perc_clip)

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax
    )

    idx_max = np.unravel_index(
        np.argmax(np.abs(rel_data)),
        rel_data.shape
    )

    slices = [
        np.fliplr(np.rot90(rel_data[idx_max[0], :, :], k=3)),
        np.rot90(rel_data[:, idx_max[1], :], k=3),
        np.rot90(rel_data[:, :, idx_max[2]], k=3)
    ]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(12, 4),
        gridspec_kw={'width_ratios': [1,1,1,0.05]}
    )

    views = [
        'Sagittal',
        'Coronal',
        'Axial'
    ]

    cmap = plt.cm.seismic

    im = None

    for ax, slc, view in zip(axes[:3], slices, views):

        im = ax.imshow(
            slc,
            cmap=cmap,
            norm=norm,
            origin='lower'
        )

        ax.set_title(view)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = plt.colorbar(im, cax=axes[3])
    cbar.set_label("Relevance")

    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# LRP
# ---------------------------------------------------------------------

def run_lrp_sfcn(
    model_path,
    dataloader,
    output_folder="SFCN_LRP",
    save_individual_nifti=False,
    save_individual_png=False,
    save_mean_map=True
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------------------------------------------
    # Modelo
    # -------------------------------------------------------------

    model = SFCN().to(device)

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    # remover prefijo "module."
    new_state_dict = {}

    for k, v in state_dict.items():

        new_key = k.replace("module.", "")

        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)

    model.eval()

    # -------------------------------------------------------------
    # Zennit
    # -------------------------------------------------------------

    composite = EpsilonPlus(epsilon=1e-6)

    attributor = Gradient(
        model,
        composite=composite
    )

    # -------------------------------------------------------------
    # Output folder
    # -------------------------------------------------------------

    os.makedirs(output_folder, exist_ok=True)

    mean_relevance_sum = None
    count = 0

    predictions = []
    reference_nii = None

    # -------------------------------------------------------------
    # Loop
    # -------------------------------------------------------------

    for idx, batch in enumerate(dataloader):

        img, age, affine, img_path = batch

        img = img.to(device)

        # ---------------------------------------------------------
        # Forward + LRP
        # ---------------------------------------------------------

        output, relevance = attributor(
            img,
            attr_output=lambda y: y
        )

        pred_age = float(
            output.detach().cpu().numpy().ravel()[0]
        )

        predictions.append(pred_age)

        # ---------------------------------------------------------
        # Relevance map
        # ---------------------------------------------------------

        rel = relevance.detach().cpu().numpy().squeeze()
        # Normalización
        rel = rel / (
            np.max(np.abs(rel)) + 1e-8
        )
        # ---------------------------------------------------------
        # Affine
        # ---------------------------------------------------------
        affine_np = affine.squeeze().numpy()
        rel_nii = nib.Nifti1Image(rel.astype(np.float32),affine_np)

        original_nii = nib.load(img_path[0])
        rel_resampled = resample_to_img(rel_nii,original_nii,interpolation="nearest")
        rel_resampled_data = rel_resampled.get_fdata()

        brain_mask = original_nii.get_fdata() > 0
        rel_resampled_data *= brain_mask

        rel_resampled_data = rel_resampled_data / (np.max(np.abs(rel_resampled_data)) + 1e-8)
        # ---------------------------------------------------------
        # Guardar NIfTI individual
        # ---------------------------------------------------------

        if save_individual_nifti:

            out_nii = os.path.join(
                output_folder,
                f"lrp_subject_{idx+1}.nii.gz"
            )

            nib.save(
                nib.Nifti1Image(
                    rel_resampled_data.astype(np.float32),
                    original_nii.affine
                ),
                out_nii
            )

        # ---------------------------------------------------------
        # Guardar PNG individual
        # ---------------------------------------------------------

        if save_individual_png:

            out_png = os.path.join(
                output_folder,
                f"lrp_subject_{idx+1}.png"
            )

            save_relevance_png_3views(
                rel_resampled_data,
                out_png
            )

        # ---------------------------------------------------------
        # Mean relevance
        # ---------------------------------------------------------

        if mean_relevance_sum is None:

            mean_relevance_sum = np.zeros_like(rel_resampled_data)

        mean_relevance_sum += rel_resampled_data

        count += 1

        print(
            f"[{count}] Predicted age: {pred_age:.2f}"
        )

    # -----------------------------------------------------------------
    # Mean map
    # -----------------------------------------------------------------

    if save_mean_map and count > 0:

        mean_rel = mean_relevance_sum / count

        # PNG
        out_png = os.path.join(
            output_folder,
            "lrp_mean_UNSAMLC_CN.png"
        )

        save_relevance_png_3views(
            mean_rel,
            out_png
        )

        # NIfTI
        mean_nii = nib.Nifti1Image(
            mean_rel.astype(np.float32),
            affine_np
        )

        out_nii = os.path.join(
            output_folder,
            "lrp_mean_UNSAMLC_CN.nii.gz"
        )

        nib.save(mean_nii, out_nii)

    return np.array(predictions)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main(csv_file, model_path):

    df = pd.read_csv(csv_file)

    dataset = MRIDataset(
        img_paths=df["Path"].values,
        ages=df["Age"].values
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    preds = run_lrp_sfcn(
        model_path=model_path,
        dataloader=dataloader
    )

    df["Predicted_Age"] = preds
    df["BAG"] = preds - df["Age"]

    #out_csv = csv_file.replace(
    #    ".csv",
    #    "_with_lrp_predictions.csv"
    #)

    #df.to_csv(out_csv, index=False)

    print("Done.")


# ---------------------------------------------------------------------
# ENTRY
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main(
        csv_file= 'ext_test_UNSAMLC_CN.csv',
        model_path= '../Training/Trained_models/model_11_sfcn.pth'
    )
