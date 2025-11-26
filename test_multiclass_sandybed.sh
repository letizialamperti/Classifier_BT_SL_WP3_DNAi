#!/bin/bash
set -e  # Termina lo script se si verifica un errore

# ----------------------------------------
# DIRETTIVE OAR (puoi adattarle)
#OAR -n test-multiclass-sandybed
#OAR -l /nodes=1/gpu=1/core=12,walltime=04:00:00
#OAR -p gpumodel='A100'
#OAR --stdout test-multiclass-sandybed-log.out
#OAR --stderr test-multiclass-sandybed-err.out
#OAR --project pr-qiepb

# Attiva l'ambiente Conda
source /applis/environments/conda.sh
conda activate zioboia

# ----------------------------------------
# PERCORSI AI DATI DI TEST
# (qui stai testando sul sottoinsieme sandybed)
# ----------------------------------------
EMBEDDINGS_FILE="BT_output/train/embedding_coords_460_all_data_.csv"
PROTECTION_FILE="label/label_protection_coralligenous.csv"
HABITAT_FILE="habitat/label_habitat_coralligenous.csv"

# Checkpoint allenato in precedenza
CHECKPOINT="CV_Class/5tdwlt2e/checkpoints/epoch=0-step=13.ckpt"

# Output CSV con le predizioni per campione
OUTPUT_CSV="test_sandy/multiclass_sandybed_test.csv"

# Crea la cartella di output se non esiste
mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "Starting MULTICLASS TEST on sandybed dataset..."
echo "Using checkpoint: $CHECKPOINT"

python test_multiclass.py \
  --embeddings_file "$EMBEDDINGS_FILE" \
  --protection_file "$PROTECTION_FILE" \
  --habitat_file    "$HABITAT_FILE" \
  --checkpoint      "$CHECKPOINT" \
  --batch_size      10 \
  --accelerator     gpu \
  --output_csv      "$OUTPUT_CSV"

echo "Test completed!"
echo "Predictions saved to: $OUTPUT_CSV"
