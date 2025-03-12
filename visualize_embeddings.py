import torch
import pandas as pd
import numpy as np
import os
import csv
from pathlib import Path
from IPython.display import display, clear_output
from ORDNA.models.barlow_twins import SelfAttentionBarlowTwinsEmbedder
from ORDNA.utils.sequence_mapper import SequenceMapper


MODEL_TYPE = 'barlow_twins' 
CHECKPOINT_PATH = Path('checkpoints/BT_2025-epoch=00.ckpt')
DATASET = '460_all_data'
SAMPLE_DIR = Path(f'/bettik/PROJECTS/pr-qiepb/lampertl')
SEQUENCE_LENGTH = 300
SAMPLE_SUBSET_SIZE = 500
NUM_CLASSES = 2 # non serve


if MODEL_TYPE == 'barlow_twins':
model = SelfAttentionBarlowTwinsEmbedder.load_from_checkpoint(CHECKPOINT_PATH, num_classes=NUM_CLASSES)
else:
raise Exception('Unknown model type:', MODEL_TYPE)

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

version = CHECKPOINT_PATH.parents[1].name
output_folder = "/BT_output"
os.makedirs(output_folder, exist_ok=True)
output_csv_file = os.path.join(output_folder, f"embedding_coords_{DATASET.lower()}_{version}.csv")



sequence_mapper = SequenceMapper()



with open(output_csv_file, mode='w', newline='') as file:
writer = csv.writer(file)
header = ["Sample", "Dim1", "Dim2", "Standard_dev_1", "Standard_dev_2"]
for class_idx in range(NUM_CLASSES):
header.append(f"Class_{class_idx}_Count")
writer.writerow(header)



num_files = len(list(SAMPLE_DIR.rglob('*.csv')))
for i, file_path in enumerate(SAMPLE_DIR.rglob('*.csv')):
    sample_name = file_path.stem
    sample_df = pd.read_csv(file_path)
    sample_df = sample_df.sample(frac=1)  # Random shuffle

    sample_emb_coords = []
    class_counts = np.zeros(NUM_CLASSES, dtype=int)
    for start in range(0, len(sample_df), SAMPLE_SUBSET_SIZE):
        end = min(start + SAMPLE_SUBSET_SIZE, len(sample_df))
        batch_df = sample_df.iloc[start:end]
        forward_sequences = batch_df['Forward'].tolist()
        reverse_sequences = batch_df['Reverse'].tolist()

        # Convert sequence data to tensor
        forward_tensor = torch.tensor(sequence_mapper.map_seq_list(forward_sequences, SEQUENCE_LENGTH))
        reverse_tensor = torch.tensor(sequence_mapper.map_seq_list(reverse_sequences, SEQUENCE_LENGTH))
        input_tensor = torch.stack((forward_tensor, reverse_tensor), dim=1).to(device)
        input_tensor = input_tensor.unsqueeze(0)  # Adjusting the batch dimension

        with torch.no_grad():
            sample_emb, class_output = model(input_tensor)
            sample_emb_coords.append(sample_emb.squeeze().cpu().numpy())
            predictions = torch.argmax(class_output.unsqueeze(0), dim=1).cpu().numpy()
            for pred in predictions:
                class_counts[pred] += 1

    if len(sample_emb_coords) > 0:
        sample_emb_coords = np.array(sample_emb_coords)
        mean_coords = np.mean(sample_emb_coords, axis=0)
        std_coords = np.std(sample_emb_coords, axis=0)
    else:
        mean_coords = np.zeros(2)
        std_coords = np.zeros(2)

    # Write results to CSV
    row = [sample_name] + list(mean_coords) + list(std_coords) + list(class_counts)
    writer.writerow(row)
    display(f'Processed {i+1}/{num_files} files')
    clear_output(wait=True)
print(f"File CSV creato con successo: {output_csv_file}")
