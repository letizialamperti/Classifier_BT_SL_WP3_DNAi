#!/bin/bash
set -e  # interrompe in caso di errore

# -------------------------------
# OAR DIRECTIVES
# -------------------------------
#OAR -n binary-test-job
#OAR -l /nodes=1/gpu=1/core=12,walltime=04:00:00
#OAR -p gpumodel='A100'
#OAR --stdout binary-test-log.out
#OAR --stderr binary-test-err.out
#OAR --project pr-qiepb

# -------------------------------
# ENVIRONMENT
# -------------------------------
source /applis/environments/conda.sh
conda activate zioboia

echo "Environment activated: zioboia"
echo "Starting binary TEST…"

# -------------------------------
# PATHS (MODIFICA QUI)
# -------------------------------
EMBEDDINGS_FILE="BT_output/train/embedding_coords_460_all_data_.csv"
PROTECTION_FILE="label/label_binaryprotection_coralligenous.csv"
HABITAT_FILE="habitat/label_habitat_coralligenous.csv"

# checkpoint scelto manualmente (SOSTITUISCI!)
CHECKPOINT="checkpoints_binary_classifier/binary-split_5_20km_sandy_bed_01-val_acc=0.89.ckpt"

# output
OUTPUT_CSV="binary_test_predictions_model3.csv"

# -------------------------------
# RUN TEST
# -------------------------------
python test_binary.py \
    --embeddings_file "$EMBEDDINGS_FILE" \
    --habitat_file "$HABITAT_FILE" \
    --protection_file "$PROTECTION_FILE" \
    --checkpoint "$CHECKPOINT" \
    --batch_size 16 \
    --output_csv "$OUTPUT_CSV"

echo "---------------------------------------"
echo "Binary TEST completed!"
echo "Output CSV → $OUTPUT_CSV"
echo "---------------------------------------"
