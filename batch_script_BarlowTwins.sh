#!/bin/bash -l

# SLURM SBATCH Directives
#SBATCH --job-name=barlow-twins-job
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --output=barlow-twins-logfile-%j.log
#SBATCH --error=barlow-twins-errorfile-%j.err
#SBATCH --account=sd29

# Module loading section
module load daint-gpu
module load cray-python

# Activate conda environment
source activate diaus_1


# Define dataset and labels path
DATASET_DIR="/bettik/PROJECTS/pr-qiepb/lampertl"
LABELS_FILE="label/ordinal_label_Sud_Corse.csv"


# Command to run the Python script
echo "Starting the training process."
srun -ul $HOME/home/lampertl/.conda/envs/zioboia/bin/python training_BarlowTwins.py \
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
