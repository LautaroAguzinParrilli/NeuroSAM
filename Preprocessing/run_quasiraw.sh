#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="/data/Lautaro/Documentos/Base_de_datos_BrainAge/NIMH_RV" # cambiar para cada base de datos
OUTPUT_BASE="/data/Lautaro/Documentos/Base_de_datos_BrainAge/preprocessing/output_preprocessing"

MASK_DIR="${OUTPUT_BASE}/masks"
QUASIRAW_DIR="${OUTPUT_BASE}/quasiraw"

mkdir -p "${MASK_DIR}" "${QUASIRAW_DIR}"

shopt -s nullglob
n=0
for nii in "${INPUT_DIR}"/*.nii.gz; do
  n=$((n+1))
  base=$(basename "${nii}")
  base_noext="${base%.nii.gz}"

  echo "--------------------------------------------"
  echo "[$n] Procesando: ${base}"

  subject_mask="${MASK_DIR}/${base_noext}_mask.nii.gz"
  subject_quasiraw="${QUASIRAW_DIR}/${base_noext}_desc-6apply_T1w.nii.gz"

  # Si ya existe el archivo final en QUASIRAW -> se omite todo
  if [ -f "${subject_quasiraw}" ]; then
    echo "  Ya procesado -> ${subject_quasiraw} (se omite)"
    continue
  fi

  # Crear archivo temporal para cerebro
  tmp_brain=$(mktemp --suffix=.nii.gz)

  # 1) Skull stripping con synthstrip si no existe la máscara
  if [ -f "${subject_mask}" ]; then
    echo "  Mask exists: ${subject_mask} (se omite synthstrip)"
  else
    echo "  Ejecutando mri_synthstrip -> solo máscara..."
    mri_synthstrip -i "${nii}" -o "${tmp_brain}" -m "${subject_mask}"
    echo "  Mask creado: ${subject_mask}"
    rm -f "${tmp_brain}"  # borrar temporal
  fi

  # 2) Brainprep quasiraw usando la máscara
  echo "  Ejecutando brainprep quasiraw..."
  brainprep quasiraw "${nii}" "${subject_mask}" "${QUASIRAW_DIR}" --no-bids
  echo "  OK: sujeto procesado -> ${QUASIRAW_DIR}"
  
  # 3) Limpiar archivos que no contienen 'desc-6apply' en su nombre
  find "${QUASIRAW_DIR}" -maxdepth 1 -type f -name "${base_noext}*" ! -name "*desc-6apply*" -exec rm -f {} \;
done

if [ "$n" -eq 0 ]; then
  echo "No se encontraron archivos *.nii.gz en ${INPUT_DIR}."
  exit 1
fi

echo "--------------------------------------------"
echo "Procesamiento completado. Masks: ${MASK_DIR}, Quasi-raw outputs: ${QUASIRAW_DIR}"


