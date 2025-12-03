#!/bin/bash
set -e  # Termina lo script se si verifica un errore

# ----------------------------------------
# DIRETTIVE OAR
#OAR -n binary-dann-460-job
#OAR -l /nodes=1/gpu=1/core=12,walltime=24:00:00
#OAR -p gpumodel='A100'
#OAR --stdout binary-dann-460-log.out
#OAR --stderr binary-dann-460-err.out
#OAR --project pr-qiepb
# ----------------------------------------

# Attiva l'ambiente Conda
source /applis/environments/conda.sh
conda activate zioboia

# Vai nella cartella del progetto
cd ~/Classifier_BT_SL_WP3_DNAi

# Percorsi ai dati GLOBALI (460 campioni, tutti gli habitat)
EMBEDDINGS_FILE="BT_output/train/embedding_coords_460_all_data_.csv"
PROTECTION_FILE="label/general_labels_numeric_binary.csv"
HABITAT_FILE="habitat/label_habitat_macro_460.csv"

echo "Starting *binary* DANN cross-validation over 5 folds on ALL 460 samples…"


LAMBDA_DOMAIN=0
LAMBDA_STR="${LAMBDA_DOMAIN/./_}"

# Cartella metriche DANN binario per questo lambda
METRICS_ROOT_DIR="metrics_binary_dann"
METRICS_DIR="${METRICS_ROOT_DIR}/lambda_${LAMBDA_STR}"
mkdir -p "$METRICS_DIR"

for fold in {1..5}; do
  fold_padded=$(printf "%02d" "$fold")
  K_CROSS_FILE="k_cross/split_5_fold_${fold_padded}.csv"

  echo
  echo "=== Fold $fold_padded: using split $K_CROSS_FILE (lambda = ${LAMBDA_DOMAIN}) ==="
  echo

  python train_binary_dann_cv.py \
    --arg_log True \
    --embeddings_file "$EMBEDDINGS_FILE" \
    --protection_file "$PROTECTION_FILE" \
    --habitat_file "$HABITAT_FILE" \
    --k_cross_file "$K_CROSS_FILE" \
    --batch_size 10 \
    --initial_learning_rate 1e-3 \
    --max_epochs 100 \
    --lambda_domain "$LAMBDA_DOMAIN" \
    --accelerator gpu

  # train_binary_dann_cv.py deve salvare:
  #   metrics_binary_dann/lambda_<LAMBDA_STR>/dann_metrics_split_5_fold_XX.csv
  METRICS_IN="${METRICS_DIR}/dann_metrics_split_5_fold_${fold_padded}.csv"
  METRICS_OUT="${METRICS_DIR}/dann_metrics_460_fold_${fold_padded}.csv"

  if [[ -f "$METRICS_IN" ]]; then
    mv "$METRICS_IN" "$METRICS_OUT"
    echo "→ Saved metrics: $METRICS_OUT"
  else
    echo "⚠️  Warning: expected metrics file $METRICS_IN not found!"
  fi

  echo "=== Fold $fold_padded completed ==="
done

echo
echo "All 5 binary DANN folds on 460 samples done!"
