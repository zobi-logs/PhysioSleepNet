"""
Wang et al. dual-stream cross-attention baseline
(arXiv 2508.02689, UbiComp/ISWC 2025 Companion).

PPG-only variant using PPG + Augmented-PPG (noise-augmented on the fly
inside the model).

Source: https://github.com/DavyWJW/sleep-staging-models
Cloned to: /data2/Akbar1/PPG_Stages/sleep-staging-models/

Harness contract (matches DeepSleepNetPPG, V8, InsightSleepNetPPG):
    forward(raw, epoch_idx, mask) -> {"main": (B, L, C_out), "latent": None}
    raw:       (B, L, C_in, T)   4-dim, pre-segmented into 30 s epochs.
                                 C_in = 1 (PPG), T = 3750 (samples at Fs=125).
    epoch_idx: (B, L)            accepted but unused (Wang uses its own PE)
    mask:      (B, L)            valid-epoch mask, True/1 = keep
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_WANG_REPO = Path("/data2/Akbar1/PPG_Stages/sleep-staging-models")
if str(_WANG_REPO) not in sys.path:
    sys.path.insert(0, str(_WANG_REPO))

from ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention  # noqa: E402


# =========================================================================
# Constants matching Wang et al.'s data protocol
# =========================================================================
HARNESS_FS      = 125            # samples per second on our side
WANG_FS         = 34.133333333   # samples per second inside Wang's model
WANG_WIN_S      = 30             # seconds per epoch
WANG_WIN_SMP    = 1024           # samples per epoch (30 s * 34.133 Hz)
WANG_WINDOWS    = 1200           # fixed 10-hour crop = 1200 epochs
WANG_TOTAL_SMP  = WANG_WINDOWS * WANG_WIN_SMP   # 1_228_800 samples


# =========================================================================
class WangDualStreamBaseline(nn.Module):
    """
    Wang et al. dual-stream cross-attention (PPG + augmented PPG).
    Fits the harness's forward(raw, epoch_idx, mask) contract.
    """

    def __init__(self,
                 num_classes: int = 4,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.inner = PPGUnfilteredCrossAttention(n_classes=num_classes,
                                                 d_model=256,
                                                 n_heads=8,
                                                 n_fusion_blocks=3)

    # ---------------------------------------------------------------------
    def _prepare_input(self, raw: torch.Tensor) -> torch.Tensor:
        """
        (B, L, C, T) at Fs=125 Hz  ->  (B, C, WANG_TOTAL_SMP) at Fs=34.133 Hz

        Wang's model expects one continuous PPG stream per subject at a
        different sample rate, padded/cropped to fixed 10-hour length.
        """
        B, L, C, T = raw.shape                            # e.g. (2, 840, 1, 3750)
        # Flatten epoch axis into time: (B, C, L*T)
        x_flat = raw.permute(0, 2, 1, 3).reshape(B, C, L * T)   # (B, 1, N_125hz)

        # Resample from 125 Hz to 34.133 Hz
        target_len = int(round(L * T * WANG_FS / HARNESS_FS))
        x_wang = F.interpolate(x_flat, size=target_len,
                               mode="linear", align_corners=False)

        # Pad or crop to Wang's fixed 10-hour length
        cur_len = x_wang.shape[-1]
        if cur_len < WANG_TOTAL_SMP:
            x_wang = F.pad(x_wang, (0, WANG_TOTAL_SMP - cur_len),
                           mode="constant", value=0.0)
        elif cur_len > WANG_TOTAL_SMP:
            x_wang = x_wang[..., :WANG_TOTAL_SMP]

        return x_wang                                      # (B, 1, 1_228_800)

    # ---------------------------------------------------------------------
    def forward(self,
                raw: torch.Tensor,
                epoch_idx=None,
                mask=None):
        """
        Args:
            raw:       (B, L, 1, T)    4-dim; each epoch is 30 s of PPG at 125 Hz
            epoch_idx: (B, L)          unused (Wang has its own positional encoding)
            mask:      (B, L)          valid-epoch mask (unused here — we return
                                       logits for all L output epochs and let
                                       the harness's loss code apply the mask)
        Returns:
            {"main": (B, L, num_classes) logits,
             "latent": None}
        """
        B, L, C, T = raw.shape

        # Bring input into Wang's expected shape / rate
        x_wang = self._prepare_input(raw)                       # (B, 1, 1_228_800)

        # Wang's model returns (B, 4, 1200) with softmax already applied
        y_soft = self.inner(x_wang)                             # (B, 4, 1200)

        # Softmax outputs -> log-probs so the harness's CE / focal losses
        # behave like on raw logits (log-softmax is what CE would produce)
        # This is safe: monotonic, and downstream argmax / metric code
        # sees the same ranking as on native logits.
        y_logits = torch.log(y_soft.clamp(min=1e-8))            # (B, 4, 1200)

        # (B, 4, 1200) -> (B, 1200, 4)
        y_logits = y_logits.transpose(1, 2)

        # Trim Wang's fixed 1200 output epochs back to L
        if y_logits.size(1) >= L:
            y_out = y_logits[:, :L, :]                          # (B, L, 4)
        else:
            # Very rare: 10-hour crop had fewer than L epochs after downsampling.
            # Pad with zeros; the harness's mask will zero these positions in
            # the loss anyway.
            pad_L = L - y_logits.size(1)
            y_out = F.pad(y_logits, (0, 0, 0, pad_L))

        return {"main": y_out, "latent": None}

    # ---------------------------------------------------------------------
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =========================================================================
# Smoke test — replicates the harness's tensor shapes
# =========================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    model = WangDualStreamBaseline(num_classes=4).cuda()

    # Simulate one recording of L epochs at Fs=125 Hz, 30 s per epoch
    B, L, C_in, T = 1, 840, 1, 3750
    raw       = torch.randn(B, L, C_in, T).cuda()
    epoch_idx = torch.arange(L).unsqueeze(0).cuda()
    mask      = torch.ones(B, L, dtype=torch.bool).cuda()

    with torch.no_grad():
        out = model(raw, epoch_idx, mask)

    print(f"raw shape       : {tuple(raw.shape)}   (B, L, C_in, T)")
    print(f"main shape      : {tuple(out['main'].shape)}   (expect (1, 840, 4))")
    print(f"latent          : {out['latent']}   (expect None)")
    print(f"params          : {model.count_parameters()/1e6:.2f} M")
    print(f"any NaN in main : {torch.isnan(out['main']).any().item()}")

    # Second smoke test — larger batch, shorter recording
    B, L = 4, 720
    raw       = torch.randn(B, L, C_in, T).cuda()
    epoch_idx = torch.arange(L).unsqueeze(0).expand(B, -1).cuda()
    mask      = torch.ones(B, L, dtype=torch.bool).cuda()
    with torch.no_grad():
        out = model(raw, epoch_idx, mask)
    print(f"\nBatch test: raw {tuple(raw.shape)} -> main {tuple(out['main'].shape)}")