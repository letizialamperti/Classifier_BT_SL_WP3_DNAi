# ORDNA/models/coral_loss_weighted_penalty.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Usiamo la tua CORAL pesata di base
from ORDNA.models.coral_loss_wheighted import coral_loss_weighted


def coral_loss_weighted_with_boundary_penalty(
    logits,
    levels,
    labels,
    num_classes,
    pos_weights=None,
    importance_weights=None,
    lambda_boundary: float = 0.05,
    reduction: str = 'mean',
):
    """
    Estende coral_loss_weighted aggiungendo una penalty MORBIDA sulla
    prima soglia CORAL (j=0), per separare meglio classe 0 da (1..K-1).

    Penalty per campione i:
        pen_i = lambda_boundary * 1[y_i>0] * (1 - sigmoid(z_{i0}))^2

    - 1[y_i>0]  → agisce solo sui campioni protetti (1..K-1)
    - (1 - sigmoid(z_{i0}))^2  → grande solo se p(y>0) è bassa
      (cioè se stiamo trattando un campione protetto come non protetto).

    Args
    ----
    logits : tensor (N, K-1)
        Logits CORAL.
    levels : tensor (N, K-1)
        Livelli CORAL 0/1 (t_ij = 1 se y_i > j).
    labels : tensor (N,)
        Etichette ordinali 0..K-1.
    num_classes : int
        K (numero di classi ordinali).
    pos_weights : tensor (K-1,), opzionale
        Pesi per i positivi per soglia.
    importance_weights : tensor (K-1,), opzionale
        Pesi di importanza per soglia.
    lambda_boundary : float
        Iperparametro che controlla la forza della penalty sul confine 0 vs >0.
        Valori consigliati: 0.01, 0.05, 0.1.
    reduction : 'mean' | 'sum' | None
        Come in coral_loss_weighted.

    Returns
    -------
    loss : tensor
        Scalare se reduction='mean'/'sum', oppure (N,) se reduction=None.
    """
    device = logits.device
    labels = labels.to(device)

    # 1) loss CORAL pesata base (per-sample)
    base_per_sample = coral_loss_weighted(
        logits,
        levels,
        pos_weights=pos_weights,
        importance_weights=importance_weights,
        reduction=None,  # -> (N,)
    )  # shape: (N,)

    # 2) Boundary penalty morbida sulla prima soglia (j=0)
    #    p0 = P(y>0) per ogni campione
    z0 = logits[:, 0]            # (N,)
    p0 = torch.sigmoid(z0)       # (N,)

    # mask: 1 per campioni protetti (classi 1..K-1), 0 per classe 0
    protected_mask = (labels > 0).float()  # (N,)

    # penalty per campione:
    #  - se y=0 => mask=0 => penalty=0
    #  - se y>0 ma p0 ~1 => (1-p0)^2 ~ 0 => penalty quasi nulla
    #  - se y>0 e p0 è bassa => penalty alta, spinge p0 verso 1
    penalty_per_sample = lambda_boundary * protected_mask * (1.0 - p0) ** 2

    total_per_sample = base_per_sample + penalty_per_sample  # (N,)

    if reduction == 'mean':
        return total_per_sample.mean()
    elif reduction == 'sum':
        return total_per_sample.sum()
    elif reduction is None:
        return total_per_sample
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class WeightedCoralLossWithBoundaryPenalty(nn.Module):
    """
    Wrapper nn.Module attorno a coral_loss_weighted_with_boundary_penalty
    da usare comodamente nel LightningModule.
    """
    def __init__(self, num_classes: int, lambda_boundary: float = 0.05, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_boundary = lambda_boundary
        self.reduction = reduction

    def forward(self, logits, levels, labels, pos_weights=None, importance_weights=None):
        return coral_loss_weighted_with_boundary_penalty(
            logits=logits,
            levels=levels,
            labels=labels,
            num_classes=self.num_classes,
            pos_weights=pos_weights,
            importance_weights=importance_weights,
            lambda_boundary=self.lambda_boundary,
            reduction=self.reduction,
        )
