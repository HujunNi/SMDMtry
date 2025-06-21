import torch
import torch.nn as nn
import torch.nn.functional as F
from mar.models.diffloss import DiffLoss

class SemanticLossWrapper(nn.Module):
    def __init__(self, loss_type="cosine", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "diffusion":
            self.loss_fn = DiffLoss(
                target_channels=kwargs.get("target_channels", 768),
                z_channels=kwargs.get("z_channels", 768),
                depth=kwargs.get("depth", 2),
                width=kwargs.get("width", 768),
                num_sampling_steps=kwargs.get("num_sampling_steps", "100"),
                grad_checkpointing=kwargs.get("grad_checkpointing", False),
            )
        elif loss_type == "cosine":
            self.loss_fn = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _cosine_loss(self, Y, z_pred, mask=None):
        # Y, z_pred: [B, K, D]
        loss = 1 - F.cosine_similarity(z_pred, Y, dim=-1)  # [B, K]
        loss_flat = loss.view(-1)
        if mask is not None:
            mask_flat = mask.view(-1).float()
            return (loss_flat * mask_flat).sum() / mask_flat.sum()
        return loss_flat.mean()

    def forward(self, Y, z_pred, mask=None):
        # Optional sanity check
        assert Y.shape == z_pred.shape, f"Shape mismatch: Y {Y.shape}, z_pred {z_pred.shape}"
        if self.loss_type == "cosine":
            return self._cosine_loss(Y, z_pred, mask)
        # diffusion branch
        B, K, D = Y.shape
        Y_flat = Y.reshape(-1, D)            # [B*K, D]
        z_flat = z_pred.reshape(-1, D)       # [B*K, D]
        mask_flat = mask.reshape(-1).float() if mask is not None else None
        return self.loss_fn(Y_flat, z_flat, mask_flat)
