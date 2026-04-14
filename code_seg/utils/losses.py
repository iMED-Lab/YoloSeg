import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss evaluated only on valid pixels.

    Expected inputs:
        logits: [B, C, H, W]
        target: [B, H, W]
        valid_mask: [B, H, W], values in {0, 1}
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        pixel_loss = self.loss_fn(logits, target.long())
        valid_mask = valid_mask.float().to(pixel_loss.device)
        masked_loss = pixel_loss * valid_mask
        return masked_loss.sum() / (valid_mask.sum() + 1e-8)


class MaskedSymmetricKLLoss(nn.Module):
    """
    Symmetric KL divergence evaluated only on valid pixels.

    Expected inputs:
        prob_a: [B, C, H, W], already softmax-normalized
        prob_b: [B, C, H, W], already softmax-normalized
        valid_mask: [B, H, W], values in {0, 1}
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        prob_a: torch.Tensor,
        prob_b: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        kl_ab = F.kl_div(
            torch.log(prob_b + 1e-8),
            prob_a.detach(),
            reduction="none",
            log_target=False,
        )
        kl_ba = F.kl_div(
            torch.log(prob_a + 1e-8),
            prob_b.detach(),
            reduction="none",
            log_target=False,
        )
        pixel_loss = 0.5 * (kl_ab + kl_ba)

        valid_mask = valid_mask.unsqueeze(1).float().to(pixel_loss.device)
        valid_mask = valid_mask.expand_as(pixel_loss)

        masked_loss = pixel_loss * valid_mask
        return masked_loss.sum() / (valid_mask.sum() + 1e-8)


def compute_consensus_mask(divergence_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert divergence mask to consensus mask.

    divergence_mask: [B, H, W], values in {0, 1}
        0 -> consensus
        1 -> divergence

    Returns:
        consensus_mask: [B, H, W], values in {0, 1}
    """
    return 1 - divergence_mask