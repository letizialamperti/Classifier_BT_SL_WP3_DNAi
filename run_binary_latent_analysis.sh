#!/bin/bash
set -e  # Exit if any command fails

# ==========================
#       OAR DIRECTIVES
# ==========================
#OAR -n coral-latent-analysis
#OAR -l /nodes=1/gpu=1/core=4,walltime=3:00:00
#OAR -p gpumodel='A100'
#OAR --stdout coral-latent-analysis.out
#OAR --stderr coral-latent-analysis.err
#OAR --project pr-qiepb

# ==========================
#    Conda environment
# ==========================
source /applis/environments/conda.sh
conda activate zioboia

# ==========================
#       INPUT FILES
# ==========================
EMBEDDINGS="BT_output/train/embedding_coords_460_all_data_.csv"
PROTECTION_FILE="label/general_labels_numeric_binary.csv"
HABITAT_FILE="habitat/empty_label_habitat_460.csv"

HABITAT_COL="habitat/label_habitat_460.csv"


# === Checkpoint da analizzare ===
CHECKPOINT="checkpoints_binary_classifier/binary-split_5_fold_01-val_acc=1.00-v21.ckpt"

# ==========================
#      ANALYSIS PARAMETERS
# ==========================
NUM_CLASSES=2
BATCH_SIZE=10
MAX_POINTS=460   # limite massimo punti per t-SNE/PCoA
WANDB_PROJECT="LatentSpaceCORAL"
WANDB_RUN_NAME="coral_weighted_latent"

echo "Running latent space analysis for checkpoint: $CHECKPOINT"

python analyze_latent_space.py \
    --checkpoint "$CHECKPOINT" \
    --embeddings_file "$EMBEDDINGS" \
    --protection_file "$PROTECTION" \
    --habitat_file "$HABITAT" \
    --habitat_labels_file "$HABITAT_COL" \
    --num_classes $NUM_CLASSES \
    --batch_size $BATCH_SIZE \
    --max_points $MAX_POINTS \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME"

echo "Latent space analysis complete!"
