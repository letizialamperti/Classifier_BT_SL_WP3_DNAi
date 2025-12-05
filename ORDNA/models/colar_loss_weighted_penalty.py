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
    CORAL loss con:
    - pos_weights: pesi per classe positiva in ogni soglia
    - importance_weights: pesi per soglia
    """
    if logits.shape != levels.shape:
        raise ValueError(f"Shape mismatch: logits {logits.shape}, levels {levels.shape}")

    log_sigmoid = F.logsigmoid(logits)
    log_one_minus_sigmoid = log_sigmoid - logits  # log(1 - sigmoid)

    t = levels

    # Termini positivi (t * log p) con pesi solo sui positivi
    if pos_weights is not None:
        weight_pos = pos_weights.view(1, -1)  # (1, K-1)
        pos_term = weight_pos * t * log_sigmoid
    else:
        pos_term = t * log_sigmoid

    # Termini negativi ((1-t) * log(1-p))
    neg_term = (1.0 - t) * log_one_minus_sigmoid

    ll = pos_term + neg_term  # log-likelihood per soglia

    # Pesi di "importanza soglia"
    if importance_weights is not None:
        iw = importance_weights.view(1, -1)
        ll = iw * ll

    per_sample_loss = -torch.sum(ll, dim=1)  # (N,)

    if reduction == 'mean':
        return per_sample_loss.mean()
    elif reduction == 'sum':
        return per_sample_loss.sum()
    elif reduction is None:
        return per_sample_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def coral_loss_weighted_with_boundary_penalty(
    logits,
    levels,
    labels,
    num_classes,
    pos_weights=None,
    importance_weights=None,
    lambda_boundary=1.0,
    reduction='mean'
):
    """
    Estende coral_loss_weighted aggiungendo una penalty che punisce
    quando un campione protetto (y>0) viene "spinto" verso la classe 0,
    usando la prima soglia CORAL (j=0).

    Penalità per campione i:
        pen_i = lambda_boundary * 1[y_i>0] * (y_i/(K-1)) * (1 - sigmoid(z_{i0}))
    """
    # 1) Loss CORAL pesata per-sample
    base_per_sample = coral_loss_weighted(
        logits,
        levels,
        pos_weights=pos_weights,
        importance_weights=importance_weights,
        reduction=None,   # <- vogliamo la loss per ogni campione
    )  # shape (N,)

    # 2) Boundary penalty sulla soglia 0
    # labels: shape (N,)
    labels = labels.to(logits.device)
    z0 = logits[:, 0]                    # prima soglia j=0
    p0 = torch.sigmoid(z0)               # p(y>0)
    protected_mask = (labels > 0).float()   # 1 se campione protetto (1..K-1)

    # normalizzazione della "forza di protezione"
    strength = labels.float() / (num_classes - 1)  # 0..1

    # penalty per campione
    penalty_per_sample = lambda_boundary * protected_mask * strength * (1.0 - p0)

    total_per_sample = base_per_sample + penalty_per_sample

    if reduction == 'mean':
        return total_per_sample.mean()
    elif reduction == 'sum':
        return total_per_sample.sum()
    elif reduction is None:
        return total_per_sample
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
