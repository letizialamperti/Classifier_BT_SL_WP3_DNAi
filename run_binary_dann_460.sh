#!/bin/bash
set -e  # Termina lo script se si verifica un errore

# ----------------------------------------
# DIRETTIVE OAR
#OAR -n binary-dann-460-job
#OAR -l /nodes=1/gpu=1/core=12,walltime=24:00:00
#OAR -p gpumodel='A100'
#OAR --stdout binary-dann-log.out
#OAR --stderr binary-dann-err.out
#OAR --project pr-qiepb
# ----------------------------------------

# Attiva l'ambiente Conda
source /applis/environments/conda.sh
conda activate zioboia

cd ~/Classifier_BT_SL_WP3_DNAi

# ----------------------------------------
# Percorsi ai dati globali (460 campioni)
# ----------------------------------------
EMBEDDINGS_FILE="BT_output/train/embedding_coords_460_all_data_.csv"
PROTECTION_FILE="label/general_labels_numeric_binary.csv"
HABITAT_FILE="habitat/label_habitat_460.csv"

echo "Starting DANN-binary cross-validation over 5 folds…"

METRICS_DIR="binary_dann_metrics"
mkdir -p "$METRICS_DIR"

for fold in {1..5}; do
  fold_padded=$(printf "%02d" "$fold")
  K_CROSS_FILE="k_cross/split_5_fold_${fold_padded}.csv"

  echo "==============================================="
  echo "=== Fold $fold_padded: split $K_CROSS_FILE ==="
  echo "==============================================="

  python train_binary_dann.py \
    --arg_log True \
    --embeddings_file "${EMBEDDINGS_FILE}" \
    --protection_file "${PROTECTION_FILE}" \
    --habitat_file "${HABITAT_FILE}" \
    --k_cross_file "${K_CROSS_FILE}" \
    --batch_size 10 \
    --initial_learning_rate 1e-3 \
    --max_epochs 100 \
    --accelerator gpu \
    --lambda_domain 4 \
    --num_classes 2 \
    --seed 42


  # train_binary_dann.py should save e.g.:
  #   binary_dann/predictions_dann_split_5_fold_XX.csv

  METRICS_IN="binary_dann/predictions_dann_split_5_fold_${fold_padded}.csv"
  METRICS_OUT="${METRICS_DIR}/binary_dann_predictions_fold_${fold_padded}.csv"

  if [[ -f "${METRICS_IN}" ]]; then
    mv "${METRICS_IN}" "${METRICS_OUT}"
    echo "→ Saved: ${METRICS_OUT}"
  else
    echo "⚠️  Warning: expected metrics file not found: ${METRICS_IN}"
  fi

  echo "=== Fold $fold_padded completed ==="
done

echo
echo "All 5 DANN binary folds completed!"
echo
