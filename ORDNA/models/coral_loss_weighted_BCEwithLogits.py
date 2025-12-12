import torch
import torch.nn.functional as F


def coral_loss_weighted(
    logits: torch.Tensor,
    levels: torch.Tensor,
    pos_weights: torch.Tensor = None,          # shape: (K-1,)
    importance_weights: torch.Tensor = None,   # shape: (K-1,)
    reduction: str = 'mean'
):
    """
    CORAL loss (BCE-with-logits) con supporto per:
    - pos_weights: come BCEWithLogitsLoss(pos_weight=...), per soglia
    - importance_weights: peso per soglia (colonna), moltiplica la loss per-colonna

    logits:  (N, K-1)
    levels:  (N, K-1) con 0/1
    """

    if logits.shape != levels.shape:
        raise ValueError(f"Shape mismatch: logits {logits.shape}, levels {levels.shape}")

    # BCE per-elemento (N, K-1), con pos_weight PyTorch-style
    # Nota: reduction='none' è essenziale per applicare importance_weights a colonna dopo
    bce = F.binary_cross_entropy_with_logits(
        logits,
        levels,
        reduction='none',
        pos_weight=pos_weights if pos_weights is not None else None
    )  # shape (N, K-1)

    # Peso per soglia (colonne)
    if importance_weights is not None:
        iw = importance_weights.view(1, -1)  # (1, K-1) broadcast su N
        bce = bce * iw

    # Somma sulle soglie -> loss per campione
    per_sample_loss = bce.sum(dim=1)  # (N,)

    # Riduzione finale
    if reduction == 'mean':
        return per_sample_loss.mean()
    elif reduction == 'sum':
        return per_sample_loss.sum()
    elif reduction is None:
        return per_sample_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class WeightedCoralLoss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        levels: torch.Tensor,
        pos_weights: torch.Tensor = None,
        importance_weights: torch.Tensor = None
    ):
        return coral_loss_weighted(
            logits,
            levels,
            pos_weights=pos_weights,
            importance_weights=importance_weights,
            reduction=self.reduction,
        )
