import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import numpy as np
import nibabel as nib
import scipy.ndimage
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img
from Model.model import build_vit3d
from Model.vit3d_pytorch import compute_attention_rollout
from Data.class_dataset import MRIDataset
from torch.utils.data import DataLoader

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

def run_attention_rollout(
    model,
    dataloader,
    device,
    output_folder="ViT_Attention_Rollout",
    save_individual_nifti=True,
    save_mean_nifti=True
):

    model.eval()

    os.makedirs(output_folder, exist_ok=True)

    mean_map = None
    count = 0

    for idx, batch in enumerate(dataloader):

        img, age, affine, img_path = batch

        img = img.to(device)

        # -------------------------------------------------
        # Forward
        # -------------------------------------------------

        with torch.no_grad():

            pred = model(img)

        # -------------------------------------------------
        # Obtener attentions
        # -------------------------------------------------

        attentions = []

        for layer in model.transformer.layers:

            attn_block = layer[0].fn.fn

            attentions.append(
                attn_block.attention_map
            )

        # -------------------------------------------------
        # Attention rollout
        # -------------------------------------------------

        rollout = compute_attention_rollout(
            attentions
        )

        # -------------------------------------------------
        # CLS token attention
        # -------------------------------------------------

        mask = rollout[:, 0, 1:]

        mask = mask.squeeze(0)

        # -------------------------------------------------
        # Reshape a grid 3D
        # -------------------------------------------------

        n_x = 176 // model.patch_size
        n_y = 208 // model.patch_size
        n_z = 176 // model.patch_size

        mask = mask.reshape(
            n_x,
            n_y,
            n_z
        )

        mask = mask.cpu().numpy()

        vmax = np.max(np.abs(mask))
        if vmax > 0:
            mask = mask / vmax

        # -------------------------------------------------
        # Upsampling a resolución MRI
        # -------------------------------------------------

        mask = scipy.ndimage.zoom(
            mask,
            (
                model.patch_size,
                model.patch_size,
                model.patch_size
            ),
            order=1
        )


        # -------------------------------------------------
        # NIfTI
        # -------------------------------------------------

        affine_np = affine.squeeze(0).numpy()

        rollout_nii = nib.Nifti1Image(
            mask.astype(np.float32),
            affine_np
        )

        original_nii = nib.load(
            img_path[0]
        )

        # -------------------------------------------------
        # Resample al espacio original
        # -------------------------------------------------

        rollout_resampled = resample_to_img(
            rollout_nii,
            original_nii,
            interpolation="nearest"
        )

        rollout_data = rollout_resampled.get_fdata()

        # -------------------------------------------------
        # Brain mask
        # -------------------------------------------------

        brain_mask = (
            original_nii.get_fdata() > 0
        )

        rollout_data *= brain_mask

        # -------------------------------------------------
        # Guardar individual
        # -------------------------------------------------
        if save_individual_nifti:

            out_nii = os.path.join(
                output_folder,
                f"attention_rollout_subject_{idx+1}.nii.gz"
            )

            #nib.save(
            #    nib.Nifti1Image(
            #        rollout_data.astype(np.float32),
            #        original_nii.affine
            #    ),
            #    out_nii
            #)
            out_png = os.path.join(
                output_folder,
                f"attention_rollout_subject_{idx+1}.png"
            )

            #save_relevance_png_3views(
            #    rollout_data,
            #    out_png
            #)

        # -------------------------------------------------
        # Mean map
        # -------------------------------------------------

        if mean_map is None:

            mean_map = np.zeros_like(
                rollout_data
            )

        mean_map += rollout_data

        count += 1

        print(
            f"[{count}] done"
        )

    # ---------------------------------------------------------
    # Mean rollout
    # ---------------------------------------------------------

    mean_map /= count

    if save_mean_nifti:
        nib.save(
            nib.Nifti1Image(
                mean_map.astype(np.float32),
                original_nii.affine
            ),
            os.path.join(
                output_folder,
                "attention_rollout_mean_UNSAMLCN.nii.gz"
            )
        )
        out_png = os.path.join(
            output_folder,
            f"attention_rollout_mean_UNSAMLCN.png"
        )

        save_relevance_png_3views(
            mean_map,
            out_png
            )

def main(csv_file, model_path):

    df = pd.read_csv(csv_file)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    model = build_vit3d()

    state_dict = torch.load(
        model_path,
        map_location=device
    )
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict)

    model=model.to(device)

    preds = run_attention_rollout(
        model=model,
        dataloader=dataloader,
        device=device,
        output_folder="ViT_Attention_Rollout"
    )

    #df["Predicted_Age"] = preds
    #df["BAG"] = preds - df["Age"]

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
        model_path= '../Training/Trained_models/model_8.pth'
    )
