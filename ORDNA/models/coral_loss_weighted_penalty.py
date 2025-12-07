# ORDNA/models/coral_loss_weighted_penalty.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ORDNA.models.coral_loss_wheighted import coral_loss_weighted


def coral_loss_weighted_with_boundary_penalty(
    logits,
    levels,
    labels,
    num_classes,
    pos_weights=None,
    importance_weights=None,
    lambda_boundary: float = 1.0,
    reduction: str = 'mean',
):
    """
    Estende coral_loss_weighted aggiungendo una penalty che punisce
    quando un campione protetto (y>0) viene "spinto" verso la classe 0,
    usando la prima soglia CORAL (j=0).

    Penalty per campione i:
        pen_i = lambda_boundary * 1[y_i>0] * (y_i/(K-1)) * (1 - sigmoid(z_{i0}))

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
    reduction : 'mean' | 'sum' | None
        Come in coral_loss_weighted.

    Returns
    -------
    loss : tensor
        Scalare se reduction='mean'/'sum', oppure (N,) se reduction=None.
    """
    device = logits.device
    labels = labels.to(device)

    # 1) loss CORAL standard pesata (per-sample)
    base_per_sample = coral_loss_weighted(
        logits,
        levels,
        pos_weights=pos_weights,
        importance_weights=importance_weights,
        reduction=None,  # -> (N,)
    )  # shape: (N,)

    # 2) Boundary penalty sulla prima soglia (j=0)
    z0 = logits[:, 0]            # (N,)
    p0 = torch.sigmoid(z0)       # p(y>0) per ogni campione

    # mask: 1 per campioni protetti (classi 1..K-1), 0 per classe 0
    protected_mask = (labels > 0).float()

    # forza di protezione normalizzata [0,1] = y/(K-1)
    strength = labels.float() / float(num_classes - 1)  # 0..1

    # penalty per campione:
    # se y=0 => mask=0 => penalty=0
    # se y>0 e p0 è basso => penalty alta
    penalty_per_sample = lambda_boundary * protected_mask * strength * (1.0 - p0)

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
    def __init__(self, num_classes: int, lambda_boundary: float = 1.0, reduction: str = 'mean'):
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
