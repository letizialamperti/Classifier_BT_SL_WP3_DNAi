import torch
import pandas as pd
from pathlib import Path


def compute_coral_weights_from_csv(protection_file: Path, num_classes: int):
    """
    Calcola:
      - importance_weights: pesi per ogni soglia CORAL (K-1)
      - pos_weights: pesi per i positivi in ogni soglia (K-1), stile BCEWithLogitsLoss(pos_weight)

    Usando la colonna 'protection' del CSV.
    """

    labels_df = pd.read_csv(protection_file)

    # 1) Conta le occorrenze di ciascuna classe e ri-indicizza su 0..num_classes-1
    counts = labels_df['protection'].value_counts().sort_index()
    counts = counts.reindex(range(num_classes), fill_value=0)   # Series di lunghezza num_classes

    # Converto in tensor per comodità
    class_counts = torch.tensor(counts.to_numpy(), dtype=torch.float32)  # shape: (K,)
    K = num_classes

    # ---------- IMPORTANCE WEIGHTS (pesi per soglia) ----------
    # Partiamo dai pesi per classe inversamente proporzionali alla frequenza
    eps = 1e-9
    cw = 1.0 / (class_counts + eps)            # (K,)
    cw = cw / cw.sum() * K                     # normalizzazione simile alla tua

    # Pesiamo ogni soglia come media dei pesi delle due classi che separa
    importance_list = [(cw[i] + cw[i+1]) / 2.0 for i in range(K-1)]
    importance_weights = torch.tensor(importance_list, dtype=torch.float32)  # (K-1,)

    # Opzionale: normalizza per avere media ~1
    importance_weights = importance_weights / importance_weights.mean()

    # ---------- POS WEIGHTS (pesi per i positivi in ogni soglia) ----------
    # Per soglia i, i positivi sono le classi > i, i negativi le classi <= i
    pos_counts = []
    neg_counts = []
    for i in range(K - 1):
        pos_count_i = class_counts[i+1:].sum()   # classi > i
        neg_count_i = class_counts[:i+1].sum()   # classi <= i
        pos_counts.append(pos_count_i)
        neg_counts.append(neg_count_i)

    pos_counts = torch.stack(pos_counts)  # (K-1,)
    neg_counts = torch.stack(neg_counts)  # (K-1,)

    # Evita divisioni per zero con clamp
    pos_weights = neg_counts / pos_counts.clamp(min=1.0)

    # Opzionale: normalizza per tenere una scala ragionevole (media ~1)
    pos_weights = pos_weights / pos_weights.mean()

    return pos_weights, importance_weights
