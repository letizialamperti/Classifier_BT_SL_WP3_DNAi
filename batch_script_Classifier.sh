#!/bin/bash -l

# SLURM SBATCH Directives
#SBATCH --job-name=classifier-job
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --output=classifier-logfile-%j.log
#SBATCH --error=classifier-errorfile-%j.err
#SBATCH --account=sd29

# Module loading section
module load daint-gpu
module load cray-python

# Activate conda environment
source activate diaus_1

# Define dataset and labels path
EMBEDDINGS_FILE="BT_output/train/embedding_coords_460_all_data_.csv"
PROTECTION_FILE="label/labels_5_levels.csv"
HABITAT_FILE="habitat/empty_label_habitat_460.csv"

# Command to run the Python script
echo "Starting the training process."
export CUDA_LAUNCH_BLOCKING=1

srun -ul $HOME/miniconda3/envs/diaus_1/bin/python training_Classifier.py \
    --arg_log True \
    --embeddings_file $EMBEDDINGS_FILE \
    --protection_file $PROTECTION_FILE \
    --habitat_file $HABITAT_FILE \
    --num_classes 5 \
    --batch_size 10 \
    --initial_learning_rate 1e-3 \
    --max_epochs 50

echo "Training completed successfully."
