import torch
import torch.nn as nn


class AbsoluteErrorLoss(nn.Module):
    """
    Loss = | pred - labels |
    - pred: valore continuo (es. output di una Linear con 1 unità)
    - labels: etichette intere (0, 1, 2, ..., num_classes-1)

    Se return_raw=True, restituisce anche il vettore degli errori assoluti
    per singolo esempio.
    """

    def __init__(self, return_raw: bool = False, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.return_raw = return_raw
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, labels: torch.Tensor, return_raw=None):
        """
        pred:   [B] o [B,1] (valore continuo)
        labels: [B] (long o float)
        """
        if return_raw is None:
            return_raw = self.return_raw

        # Se pred ha shape [B,1], schiacciala a [B]
        if pred.dim() == 2 and pred.size(1) == 1:
            pred = pred.squeeze(1)

        # Converti le labels a float per la sottrazione
        labels = labels.float()

        abs_err = torch.abs(pred - labels)

        if self.reduction == "mean":
            loss = abs_err.mean()
        elif self.reduction == "sum":
            loss = abs_err.sum()
        else:  # "none"
            loss = abs_err

        if return_raw:
            # ritorniamo anche il vettore degli errori assoluti (per esempio per istogrammi)
            return loss, abs_err

        return loss


def regression_to_label(pred: torch.Tensor, num_classes: int):
    """
    Converte una predizione continua in una label discreta:
    - arrotonda
    - clamp a [0, num_classes-1]
    """
    if pred.dim() == 2 and pred.size(1) == 1:
        pred = pred.squeeze(1)

    pred_rounded = torch.round(pred).long()
    pred_clamped = torch.clamp(pred_rounded, 0, num_classes - 1)
    return pred_clamped
