#!/bin/bash
set -e  # Termina lo script se si verifica un errore

# ----------------------------------------
# DIRETTIVE OAR
# Nota: le direttive OAR devono iniziare con "#OAR"
# Modifica i parametri in base alle risorse e alle policy del tuo cluster
#Esempio con OAR:
#OAR -n classifier-job
#OAR -l /nodes=1/gpu=1/core=12,walltime=24:00:00
#OAR -p gpumodel='A100'
#OAR --stdout classifier-logfile.out
#OAR --stderr classifier-errorfile.err
#OAR --project pr-qiepb


# Attiva l'ambiente Conda (modifica il percorso e il nome dell'ambiente se necessario)
source /applis/environments/conda.sh
conda activate zioboia

# Definisci i percorsi per i file dei dati
EMBEDDINGS_FILE="BT_output/train/embedding_coords_460_all_data_.csv"
PROTECTION_FILE="label/labels_5_levels.csv"
HABITAT_FILE="habitat/empty_label_habitat_460.csv"

# Avvia il training del classificatore
echo "Starting the training process."
export CUDA_LAUNCH_BLOCKING=1


    python train_cv.py \
  --embeddings_file $EMBEDDINGS_FILE \
  --protection_file $PROTECTION_FILE \
  --habitat_file $HABITAT_FILE \
  --k_cross_file k_cross/split_5_fold_01.csv \
  --num_classes 5 \
  --batch_size 10 \
  --initial_learning_rate 1e-3 \
  --max_epochs 100 \
  --arg_log True

echo "Training completed successfully."
