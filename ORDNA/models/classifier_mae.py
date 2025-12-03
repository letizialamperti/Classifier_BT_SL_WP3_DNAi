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

        labels_long = labels.long()
        labels_float = labels.float()

        abs_err = torch.abs(pred - labels_float)  # [B]

        # --- pesatura per classe ---
        if self.class_weights is not None:
            # weights per esempio = 1/freq
            sample_weights = self.class_weights[labels_long]  # [B]
            abs_err = abs_err * sample_weights

        if self.reduction == "mean":
            loss = abs_err.mean()
        elif self.reduction == "sum":
            loss = abs_err.sum()
        else:  # "none"
            loss = abs_err

        if return_raw:
            # ritorniamo il vettore di errori pesati (per istogrammi ecc.)
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
