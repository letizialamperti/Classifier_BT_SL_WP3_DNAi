import torch
import torch.nn.functional as F


def coral_loss_weighted(
    logits,
    levels,
    pos_weights=None,          # shape: (num_classes-1,)
    importance_weights=None,   # shape: (num_classes-1,)
    reduction='mean'
):
    """
    CORAL loss con supporto per:
    - pos_weights: pesi per la classe positiva in ogni soglia (tipo BCEWithLogitsLoss(pos_weight))
    - importance_weights: pesi per ogni soglia (colonna)
    
    logits: tensor (N, K-1)
    levels: tensor (N, K-1) con 0/1
    """

    if logits.shape != levels.shape:
        raise ValueError(f"Shape mismatch: logits {logits.shape}, levels {levels.shape}")

    # log_sigmoid(z) = log(sigmoid(z))
    log_sigmoid = F.logsigmoid(logits)
    # log(1 - sigmoid(z)) = log_sigmoid(-z) = log_sigmoid(z) - z
    log_one_minus_sigmoid = log_sigmoid - logits

    # t, z, p come da BCE
    t = levels
    # parte positiva: t * log(p)
    # parte negativa: (1-t) * log(1-p)
    # introduciamo pos_weights per i positivi
    if pos_weights is not None:
        # pos_weights shape: (K-1,) -> broadcast su batch dim
        # weight_pos[j] agisce su tutti i positivi nella colonna j
        weight_pos = pos_weights.view(1, -1)
        pos_term = weight_pos * t * log_sigmoid
    else:
        pos_term = t * log_sigmoid

    neg_term = (1.0 - t) * log_one_minus_sigmoid

    # log-likelihood (positivo): pos_term + neg_term
    ll = pos_term + neg_term  # shape (N, K-1)

    # importance_weights per soglia (colonne)
    if importance_weights is not None:
        iw = importance_weights.view(1, -1)  # (1, K-1)
        ll = iw * ll

    # loss = - somma su soglie
    per_sample_loss = -torch.sum(ll, dim=1)  # shape (N,)

    if reduction == 'mean':
        return per_sample_loss.mean()
    elif reduction == 'sum':
        return per_sample_loss.sum()
    elif reduction is None:
        return per_sample_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class WeightedCoralLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, levels, pos_weights=None, importance_weights=None):
        return coral_loss_weighted(
            logits,
            levels,
            pos_weights=pos_weights,
            importance_weights=importance_weights,
            reduction=self.reduction,
        )
