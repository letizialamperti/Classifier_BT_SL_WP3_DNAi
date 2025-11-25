#!/bin/bash
set -e  # Termina lo script se si verifica un errore

# ----------------------------------------
# DIRETTIVE OAR (modifica a piacere)
#OAR -n classifier-habitat-job
#OAR -l /nodes=1/gpu=1/core=12,walltime=24:00:00
#OAR -p gpumodel='A100'
#OAR --stdout classifier-sandybed-log.out
#OAR --stderr classifier-sandybed-err.out
#OAR --project pr-qiepb

# Attiva l'ambiente Conda
source /applis/environments/conda.sh
conda activate zioboia

# Percorsi ai dati (EMBEDDINGS e PROTECTION restano invariati)
EMBEDDINGS_FILE="BT_output/train/embedding_coords_460_all_data_.csv"
PROTECTION_FILE="label/label_protection_sandybed.csv"
HABITAT_FILE="habitat/label_habitat_sandybed.csv"

echo "Starting habitat-aware cross-validation over 5 folds…"

for fold in {1..5}; do
  fold_padded=$(printf "%02d" "$fold")
  K_CROSS_FILE="k_cross/split_5_sandy_bed_${fold_padded}.csv"
  echo "=== Fold $fold_padded: using split $K_CROSS_FILE and habitat $HABITAT_FILE ==="

  # Avvia il training sul fold corrente
  python train_cv.py \
    --arg_log True \
    --embeddings_file "$EMBEDDINGS_FILE" \
    --protection_file "$PROTECTION_FILE" \
    --habitat_file "$HABITAT_FILE" \
    --k_cross_file "$K_CROSS_FILE" \
    --num_classes 5 \
    --batch_size 10 \
    --initial_learning_rate 1e-3 \
    --max_epochs 100 \
    --accelerator gpu

  # Rinomina il CSV delle metriche aggiungendo il suffisso _habitat
  METRICS_IN="metrics_sandybed_5_fold_${fold_padded}.csv"
  METRICS_OUT="metrics_sandybed_5_fold_${fold_padded}.csv"
  if [[ -f "$METRICS_IN" ]]; then
    mv "$METRICS_IN" "$METRICS_OUT"
    echo "→ Renamed $METRICS_IN to $METRICS_OUT"
  else
    echo "⚠️  Warning: expected metrics file $METRICS_IN not found!"
  fi

  echo "=== Fold $fold_padded completed ==="
done

echo "All 5 with-habitat sandybed multiclass folds done!"
