#!/bin/bash

#OAR -n barlow-twins-job

#OAR -l /nodes=1/core=12,walltime=24:00:00

#OAR -p gpu_count>0

#OAR --stdout barlow-twins-logfile.out
#OAR --stderr barlow-twins-errorfile.err

#OAR --project pr-qiepb


cd /bettik/PROJECTS/pr-qiepb/lampertl

# Attivare Conda
source /applis/environments/conda.sh
conda activate zioboia

# Definire percorsi dataset e labels
DATASET_DIR="/bettik/PROJECTS/pr-qiepb/lampertl"
LABELS_FILE="label/labels_5_levels.csv"

# Eseguire lo script Python
echo "Starting the training process."
python training_BarlowTwins.py \
    --arg_log True \
    --samples_dir $DATASET_DIR \
    --labels_file $LABELS_FILE \
    --embedder_type barlow_twins \
    --sequence_length 300 \
    --sample_subset_size 500 \
    --num_classes 5 \
    --batch_size 32 \
    --token_emb_dim 8 \
    --sample_repr_dim 64 \
    --sample_emb_dim 2 \
    --barlow_twins_lambda 1 \
    --initial_learning_rate 1e-3 \
    --max_epochs 1

echo "Training completed successfully."
