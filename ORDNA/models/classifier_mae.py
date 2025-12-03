# classifier_mae.py

import torch
import torch.nn as nn


class AbsoluteErrorLoss(nn.Module):
    """
    Loss pesata:
      loss_i = w[label_i] * |pred_i - label_i|

    - pred:   valore continuo ([B] o [B,1])
    - labels: etichette intere in [0, num_classes-1]
    - class_weights: tensor di shape [num_classes] (opzionale)
    """

    def __init__(
        self,
        return_raw: bool = False,
        reduction: str = "mean",
        num_classes: int | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.return_raw = return_raw
        self.reduction = reduction

        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            if num_classes is not None and cw.numel() != num_classes:
                raise ValueError(
                    f"class_weights length ({cw.numel()}) does not match num_classes ({num_classes})"
                )
            # buffer → si muove automaticamente su cuda/cpu con .to(device)
            self.register_buffer("class_weights", cw)
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, labels: torch.Tensor, return_raw=None):
        if return_raw is None:
            return_raw = self.return_raw

        # Se pred ha shape [B,1], schiacciala a [B]
        if pred.dim() == 2 and pred.size(1) == 1:
            pred = pred.squeeze(1)

        labels_long  = labels.long()
        labels_float = labels.float()

        # errore
        diff    = pred - labels_float           # [B]
        abs_err = torch.abs(diff)               # |pred - label|
        sq_err  = diff ** 2                     # (pred - label)^2

        # loss base per-sample: |e| + e^2
        per_sample_loss = abs_err + sq_err      # [B]

        # --- pesatura per classe ---
        if self.class_weights is not None:
            sample_weights = self.class_weights[labels_long]   # [B]
            per_sample_loss = per_sample_loss * sample_weights

        # riduzione
        if self.reduction == "mean":
            loss = per_sample_loss.mean()
        elif self.reduction == "sum":
            loss = per_sample_loss.sum()
        else:  # "none"
            loss = per_sample_loss

        if return_raw:
            # ritorniamo il vettore di loss per-sample (già pesato)
            return loss, per_sample_loss

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
