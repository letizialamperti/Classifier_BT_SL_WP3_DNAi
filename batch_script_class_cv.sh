#!/bin/bash
set -e  # Termina lo script se si verifica un errore

# ----------------------------------------
# DIRETTIVE OAR (modifica a piacere)
#OAR -n classifier-job
#OAR -l /nodes=1/gpu=1/core=12,walltime=24:00:00
#OAR -p gpumodel='A100'
#OAR --stdout classifier-logfile.out
#OAR --stderr classifier-errorfile.err
#OAR --project pr-qiepb

# Attiva l'ambiente Conda
source /applis/environments/conda.sh
conda activate zioboia

# Percorsi ai dati (non cambiano fold-per-fold)
EMBEDDINGS_FILE="BT_output/train/embedding_coords_460_all_data_.csv"
PROTECTION_FILE="label/labels_5_levels.csv"
HABITAT_FILE="habitat/empty_label_habitat_460.csv"

echo "Starting cross-validation over 5 folds…"

for fold in {1..5}; do
  fold_padded=$(printf "%02d" "$fold")
  K_CROSS_FILE="k_cross/split_5_fold_${fold_padded}.csv"

  echo "=== Fold $fold_padded → using split file $K_CROSS_FILE ==="

  export CUDA_LAUNCH_BLOCKING=1

  python train_cv_multiclass_coralwheighted.py \
    --arg_log True \
    --embeddings_file "$EMBEDDINGS_FILE" \
    --protection_file "$PROTECTION_FILE" \
    --habitat_file "$HABITAT_FILE" \
    --k_cross_file "$K_CROSS_FILE" \
    --num_classes 5 \
    --batch_size 32 \
    --initial_learning_rate 1e-3 \
    --max_epochs 100 \
    --accelerator gpu

  echo "=== Fold $fold_padded completed ==="
done

echo "All 5 folds have been processed successfully."
