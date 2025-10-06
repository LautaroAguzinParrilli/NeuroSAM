#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="/data/Lautaro/Documentos/Base_de_datos_BrainAge/preprocessing/output_preprocessing/quasiraw"
OUTPUT_DIR="/data/Lautaro/Documentos/Base_de_datos_BrainAge/preprocessing/quality_control_batches"

BATCH_SIZE=100
CORR_THR=0.5
MASK_REGEX="None"

mkdir -p "${OUTPUT_DIR}"

all_imgs=(${INPUT_DIR}/*_desc-6apply_T1w.nii.gz)
total=${#all_imgs[@]}

echo "Total de imágenes encontradas: $total"
echo "Procesando en lotes de $BATCH_SIZE..."

batch=0
for ((i=0; i<total; i+=BATCH_SIZE)); do
    batch=$((batch+1))
    batch_dir="${OUTPUT_DIR}/batch_${batch}"
    mkdir -p "${batch_dir}"

    tmp_batch="${batch_dir}/tmp_imgs"
    mkdir -p "${tmp_batch}"

    subset=("${all_imgs[@]:i:BATCH_SIZE}")
    echo "--------------------------------------"
    echo "Batch ${batch}: procesando ${#subset[@]} sujetos"
    echo "Output en: ${batch_dir}"

    # copiar subset a carpeta temporal
    for f in "${subset[@]}"; do
        ln -s "$f" "${tmp_batch}/"  # link simbólico para ahorrar espacio
    done

    # correr brainprep en la carpeta temporal
    if [ "${MASK_REGEX}" = "None" ]; then
        brainprep quasiraw-qc "${tmp_batch}/*.nii.gz" "${batch_dir}" --corr_thr "${CORR_THR}"
    else
        brainprep quasiraw-qc "${tmp_batch}/*.nii.gz" "${batch_dir}" --brainmask_regex "${MASK_REGEX}" --corr_thr "${CORR_THR}"
    fi

    # limpiar links temporales
    rm -rf "${tmp_batch}"
done
