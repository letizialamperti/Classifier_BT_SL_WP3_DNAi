#!/bin/bash
set -e  # Exit if any command fails

# ==========================
#       OAR DIRECTIVES
# ==========================
#OAR -n tsne-dann-viz
#OAR -l /nodes=1/gpu=1/core=4,walltime=2:00:00
#OAR -p gpumodel='A100'
#OAR --stdout test_binary_script.out
#OAR --stderr test_binary_script.err
#OAR --project pr-qiepb

# ==========================
#    Conda environment
# ==========================
source /applis/environments/conda.sh
conda activate zioboia

# ==========================
#       INPUT FILES
# ==========================
EMBEDDINGS_FILE="BT_output/train/embedding_coords_460_all_data_.csv"
PROTECTION_FILE="label/general_labels_numeric_binary.csv"
HABITAT_FILE="habitat/empty_label_habitat_460.csv"

# === Checkpoint da testare ===
CHECKPOINT="checkpoints_binary_classifier/binary-new_no_coral_split_4_fold_01-val_acc=0.95.ckpt"
BATCH_SIZE=32

# ==========================
#   SPYGEN CODES TARGET
# ==========================
SPYGEN_CODES=(
  "SPY211052" "SPY211058" "SPY231897" "SPY231898" "SPY231899" "SPY231900" "SPY231903"
  "SPY231904" "SPY212588" "SPY212589" "SPY231885" "SPY231886" "SPY231891" "SPY231892"
  "SPY211069" "SPY211070" "SPY211045" "SPY211046" "SPY211049" "SPY211050" "SPY231909"
  "SPY231910" "SPY211056" "SPY211064" "SPY231911" "SPY231912" "SPY231893" "SPY231894"
)

python test_binary_classifer.py \
    --embeddings_file "$EMBEDDINGS_FILE" \
    --protection_file "$PROTECTION_FILE" \
    --habitat_file "$HABITAT_FILE" \
    --checkpoint "$CHECKPOINT" \
    --batch_size "$BATCH_SIZE" \
    --device cuda \
    --spygen_codes "${SPYGEN_CODES[@]}" \
    --output_csv "test_binary_coralligenous_spygen_codes.csv"

echo "Test classifier completato per i campioni coralligenous."
