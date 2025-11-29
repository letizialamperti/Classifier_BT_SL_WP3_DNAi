#!/bin/bash
set -e  # Termina lo script in caso di errore

# ----------------------------------------
# DIRETTIVE OAR
#OAR -n dann-sandybed-job
#OAR -l /nodes=1/gpu=1/core=12,walltime=24:00:00
#OAR -p gpumodel='A100'
#OAR --stdout dann-sandybed-log.out
#OAR --stderr dann-sandybed-err.out
#OAR --project pr-qiepb
# ----------------------------------------

echo "→ Starting DANN CORAL multiclass training (Sandy Bed)…"

# Attiva ambiente Conda
source /applis/environments/conda.sh
conda activate zioboia

# ---------------------------------------------------------
# PERCORSI DEI DATI (versione Sandy Bed)
# ---------------------------------------------------------
EMBEDDINGS_FILE="BT_output/train/embedding_coords_460_all_data_.csv"
PROTECTION_FILE="label/label_protection_sandybed.csv"
HABITAT_FILE="habitat/label_habitat_sandybed.csv"

# Splits creati con blockCV (es. 20 km)
SPLIT_PREFIX="k_cross/split_5_20km_sandy_bed_"

# Output delle metriche
METRICS_DIR="metrics_dann"
mkdir -p "$METRICS_DIR"

# ---------------------------------------------------------
# LOOP SUI 5 FOLD
# ---------------------------------------------------------

for fold in {1..5}; do
    fold_padded=$(printf "%02d" "$fold")
    K_CROSS_FILE="${SPLIT_PREFIX}${fold_padded}.csv"

    echo
    echo "=============================================="
    echo "  Running fold ${fold_padded} using file:"
    echo "  → ${K_CROSS_FILE}"
    echo "=============================================="

    python train_dann_cv.py \
        --arg_log True \
        --embeddings_file "$EMBEDDINGS_FILE" \
        --protection_file "$PROTECTION_FILE" \
        --habitat_file "$HABITAT_FILE" \
        --k_cross_file "$K_CROSS_FILE" \
        --num_classes 5 \
        --batch_size 10 \
        --initial_learning_rate 1e-3 \
        --max_epochs 100 \
        --lambda_domain 1.0 \
        --accelerator gpu

    # Aspettato: metrics_dann/dann_metrics_split_5_20km_sandy_bed_XX.csv
    METRICS_IN="${METRICS_DIR}/dann_metrics_split_5_20km_sandy_bed_${fold_padded}.csv"
    METRICS_OUT="${METRICS_DIR}/dann_metrics_sandybed_${fold_padded}.csv"

    if [[ -f "$METRICS_IN" ]]; then
        mv "$METRICS_IN" "$METRICS_OUT"
        echo "→ Saved metrics: $METRICS_OUT"
    else
        echo "⚠️  Warning: expected metrics file not found:"
        echo "   $METRICS_IN"
    fi

    echo "=== Fold $fold_padded DONE ==="
done

echo
echo "All 5 DANN sandy bed folds completed successfully!"
