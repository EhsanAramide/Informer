import torch

class TriangularCausalMask():
    """Boolean causal mask shaped (B, 1, L, L), with true meaning blocked."""
    def __init__(self, B, L, device="cpu"):
        """Construct a mask for batch size B and length L on the given device."""
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """Return the stored Boolean tensor directly (without copying)."""
        return self._mask

class ProbMask():
    """Causal mask for the selected ProbSparse query rows."""
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """Select query rows using index shaped (B, H, selected_queries).

        B/H/L are batch/head/full-query counts; scores supplies output shape
        and key length. index and scores must be compatible with device.
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        """Return the stored Boolean mask with the same shape as scores."""
        return self._mask
