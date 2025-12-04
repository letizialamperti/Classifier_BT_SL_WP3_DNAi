def compute_coral_weights(y_train: torch.Tensor, num_classes: int):
    """
    y_train: tensor (N,) con etichette 0..num_classes-1
    ritorna:
      - pos_weights: tensor (num_classes-1,)
      - importance_weights: tensor (num_classes-1,)
    """
    K = num_classes

    pos_counts = []
    neg_counts = []

    for i in range(K - 1):
        pos_mask = (y_train > i)   # target 1 per soglia i
        neg_mask = ~pos_mask       # target 0 per soglia i

        pos_counts.append(pos_mask.sum().float())
        neg_counts.append(neg_mask.sum().float())

    pos_counts = torch.stack(pos_counts)  # (K-1,)
    neg_counts = torch.stack(neg_counts)  # (K-1,)

    # -------- pos_weights: bilancia pos vs neg in ogni soglia ----------
    pos_weights = neg_counts / pos_counts.clamp(min=1.0)
    # opzionale: normalizziamo per non avere pesi enormi
    pos_weights = pos_weights / pos_weights.mean()

    # -------- importance_weights: peso relativo tra soglie ----------
    importance_weights = (1.0 / pos_counts.clamp(min=1.0)).sqrt()
    importance_weights = importance_weights / importance_weights.mean()

    return pos_weights, importance_weights
