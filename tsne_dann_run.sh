#!/bin/bash
set -e  # Exit if any command fails

# ==========================
#       OAR DIRECTIVES
# ==========================
#OAR -n tsne-dann-viz
#OAR -l /nodes=1/gpu=1/core=4,walltime=2:00:00
#OAR -p gpumodel='A100'
#OAR --stdout tsne-dann-viz.out
#OAR --stderr tsne-dann-viz.err
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
PROTECTION="label/labels_5_levels.csv"
HABITAT="habitat/label_habitat_460.csv"

# === Choose the checkpoint you want to visualize ===
CHECKPOINT="checkpoints_dann_classifier/lambda_0_5/split_5_fold_03-best.ckpt"

# ==========================
#      t-SNE ARGUMENTS
# ==========================
NUM_CLASSES=5
BATCH_SIZE=64
PERPLEXITY=30
LAMBDA_DOMAIN=0.5

echo "Running t-SNE visualization for checkpoint: $CHECKPOINT"

python visualize_dann_tsne.py \
    --embeddings_file "$EMBEDDINGS" \
    --protection_file "$PROTECTION" \
    --habitat_file "$HABITAT" \
    --checkpoint "$CHECKPOINT" \
    --num_classes $NUM_CLASSES \
    --batch_size $BATCH_SIZE \
    --perplexity $PERPLEXITY \
    --lambda_domain $LAMBDA_DOMAIN \
    --device cuda \
    --output_prefix "tsne_fold03_lambda05"

echo "t-SNE visualization complete!"
