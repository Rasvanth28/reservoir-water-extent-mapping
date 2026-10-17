"""Losses. Labels use 255 = ignore (unlabeled, invalid, or within the validation-point exclusion)."""
IGNORE_INDEX = 255


def masked_ce(logits, target, weight=None):
    """Cross-entropy over [B, C, H, W] logits and [B, H, W] int targets, ignoring 255.

    Returns 0 (with a graph) when a batch has no labeled pixel, instead of NaN.
    """
    import torch.nn.functional as F
    loss = F.cross_entropy(logits, target, weight=weight, ignore_index=IGNORE_INDEX, reduction="sum")
    n = (target != IGNORE_INDEX).sum().clamp(min=1)
    return loss / n


def per_date_ce(logits, mask, weight=None):
    """logits [B, T, C, H, W], mask [B, T, H, W] -> mean of the per-date masked CEs."""
    T = logits.shape[1]
    return sum(masked_ce(logits[:, t], mask[:, t], weight) for t in range(T)) / T
