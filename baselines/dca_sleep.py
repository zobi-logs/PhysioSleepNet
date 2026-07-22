"""
DCA-Sleep baseline adapter — faithful reimplementation.

Paper: Li, He, Bie, Guo, Wang, Zhao, Zhong, Zhu.
       "Large-Scale Validation of a Dual Cross-Attention Network for
        Automated Sleep Staging Using Wearable Photoplethysmography Signals."
       Diagnostics 16(5):802, March 2026.
       doi: 10.3390/diagnostics16050802

Architecture (per Section 2 of the paper):
    - Fully convolutional U-Net-style network with 12 encoder + 12 decoder
      blocks and skip connections at every level
    - Dual Cross-Attention (DCA) module in the middle: CCA (channel) + TCA
      (temporal), operating on multi-scale encoder features (Eqs. 1-9)
    - Input: 40 consecutive 30 s epochs at 128 Hz = 153,600 samples/window
    - Output: (40, 4) — one prediction per 30 s epoch
    - Kernel size 9 for all conv layers; downsampling via MaxPool k=2 s=2

Cleanest framing (matches our SleepPPG-Net convention):
    We include DCA-Sleep architecturally to represent the most recent iteration
    of this family; cross-modality ECG pretraining was not applied so that all
    baselines share our PPG-only, five-fold-CV protocol on MESA and CFS.

Harness contract:
    forward(raw, epoch_idx, mask) -> {"main": (B, L, C), "latent": None}
    raw:       (B, L, 1, T)   4-dim, T = 3750 samples per epoch at Fs=125 Hz
    epoch_idx: (B, L)         accepted but unused
    mask:      (B, L)         accepted but unused (harness applies to loss)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Constants matching the paper's protocol
# =========================================================================
HARNESS_FS       = 125           # samples per second on our side
DCA_FS           = 128           # DCA-Sleep's internal rate
EPOCH_S          = 30            # seconds per epoch
DCA_EPOCH_SMP    = EPOCH_S * DCA_FS         # 3840 samples per epoch @ 128 Hz
WINDOW_EPOCHS    = 40            # 40 epochs per window (20 min)
WINDOW_SMP_RAW   = WINDOW_EPOCHS * DCA_EPOCH_SMP     # 153,600 samples/window
# Pad to next multiple that gives clean 2^N downsampling to 40 tokens:
# 40 * 4096 = 163,840 (4096 = 2^12)
WINDOW_SMP_PAD   = WINDOW_EPOCHS * 4096              # 163,840

# Encoder / decoder channel schedule (12 blocks):
# Grow through 5 doublings, then plateau at 256 for the remaining 7 blocks.
# This keeps parameter count reasonable (~15-25 M) while matching the paper's
# 12-block depth.
CHANNELS = [16, 32, 64, 128, 256, 256, 256, 256, 256, 256, 256, 256]
assert len(CHANNELS) == 12


# =========================================================================
# ENCODER BLOCK — Conv1D + ELU + BatchNorm + MaxPool (downsample /2)
# =========================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=pad)
        self.elu = nn.ELU(inplace=True)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.elu(x)
        x = self.bn(x)
        skip = x                             # save PRE-pool for skip connection
        x = self.pool(x)
        return x, skip


# =========================================================================
# DECODER BLOCK — BatchNorm + Conv1D + ELU + Upsample (upsample x2)
# =========================================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 9,
                 upsample: bool = True):
        super().__init__()
        pad = kernel_size // 2
        self.bn = nn.BatchNorm1d(in_ch)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=pad)
        self.elu = nn.ELU(inplace=True)
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False) \
                  if upsample else nn.Identity()

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.elu(x)
        # PyTorch 1.10.x has no bf16 kernel for linear upsample.
        # Cast to fp32 around the upsample, cast back to input dtype.
        if isinstance(self.up, nn.Upsample):
            orig_dtype = x.dtype
            with torch.cuda.amp.autocast(enabled=False):
                x = self.up(x.float())
            x = x.to(orig_dtype)
        else:
            x = self.up(x)
        return x


# =========================================================================
# DCA MODULE — implements Eqs. 1-9 from the paper
#
# For each of the N encoder scales, we:
#   1) AdaptiveAvgPool1d(L) + depthwise conv        (Eq. 1)  -> E_i (B, C_i, L)
#   2) Concat across scales along channel dim       (Eq. 2)  -> E_c (B, d_c, L)
#   3) CCA: attention over CHANNELS                 (Eqs. 3-5) -> X_i (B, C_i, L)
#   4) TCA: attention over TIME                     (Eqs. 6-8) -> Y_i (B, C_i, L)
#   5) F_out_i = X_i + Y_i                          (Eq. 9)
#
# Refined features are then interpolated back to each scale's original
# temporal length and added residually to the original skip connections.
#
# Design note: parameters are kept modest via a common projection dim d_proj
# to prevent quadratic explosion at d_c ~ 2500. This preserves the paper's
# design philosophy while keeping the param count reasonable.
# =========================================================================
class DCAModule(nn.Module):
    def __init__(self, encoder_channels, L: int = WINDOW_EPOCHS,
                 d_proj: int = 128, n_heads: int = 4):
        super().__init__()
        self.L = L
        self.encoder_channels = encoder_channels
        self.n_scales = len(encoder_channels)
        self.d_proj = d_proj
        self.n_heads = n_heads
        self.d_c = d_proj * self.n_scales        # global context dim
        assert self.d_c % n_heads == 0
        self.d_k = self.d_c // n_heads

        # Per-scale: adaptive pool -> depthwise conv -> project to d_proj
        self.per_scale_prep = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(L),
                nn.Conv1d(c, c, kernel_size=3, padding=1, groups=c),  # depthwise
                nn.Conv1d(c, d_proj, kernel_size=1),                  # pointwise
            ) for c in encoder_channels
        ])

        # --- CCA sub-module (Eqs. 3-5) ---
        # Q_i from E_i (per scale), K_c and V_c from E_c (shared)
        self.cca_q = nn.ModuleList([
            nn.Conv1d(d_proj, d_proj, kernel_size=1) for _ in encoder_channels
        ])
        self.cca_k = nn.Conv1d(self.d_c, self.d_c, kernel_size=1)
        self.cca_v = nn.Conv1d(self.d_c, self.d_c, kernel_size=1)
        # Project attention output back to each scale's channel count
        self.cca_out = nn.ModuleList([
            nn.Conv1d(self.d_c, c, kernel_size=1) for c in encoder_channels
        ])
        # LayerNorm over channel dim (Eq. 5)
        self.cca_ln = nn.ModuleList([nn.LayerNorm(c) for c in encoder_channels])

        # --- TCA sub-module (Eqs. 6-8) ---
        # Q_t, K_t, V_t all from E_c
        self.tca_q = nn.Conv1d(self.d_c, self.d_c, kernel_size=1)
        self.tca_k = nn.Conv1d(self.d_c, self.d_c, kernel_size=1)
        self.tca_v = nn.Conv1d(self.d_c, self.d_c, kernel_size=1)
        # Project attention output back to each scale's channel count
        self.tca_out = nn.ModuleList([
            nn.Conv1d(self.d_c, c, kernel_size=1) for c in encoder_channels
        ])
        # LayerNorm over channel dim (Eq. 8)
        self.tca_ln = nn.ModuleList([nn.LayerNorm(c) for c in encoder_channels])

    # ---------------------------------------------------------------------
    def forward(self, skips):
        """
        skips: list of (B, C_i, T_i) — encoder features at each scale (pre-pool)
        Returns: list of (B, C_i, T_i) — DCA-enhanced skip features
        """
        # -- Step 1: per-scale pool + depthwise conv -> E_i (B, d_proj, L)
        E_list = [prep(s) for prep, s in zip(self.per_scale_prep, skips)]

        # -- Step 2: concat along channel dim -> E_c (B, d_c, L)
        E_c = torch.cat(E_list, dim=1)                           # (B, d_c, L)

        # -- CCA: attention where channels are the "sequence" --
        Kc = self.cca_k(E_c)                                     # (B, d_c, L)
        Vc = self.cca_v(E_c)                                     # (B, d_c, L)
        # Reshape for attention over channel dim: (B, L, d_c) treats channels as tokens
        Kc_t = Kc.transpose(1, 2)                                # (B, L, d_c)
        Vc_t = Vc.transpose(1, 2)                                # (B, L, d_c)

        cca_outs = []
        for i, (E_i, q_proj, out_proj, ln) in enumerate(
            zip(E_list, self.cca_q, self.cca_out, self.cca_ln)
        ):
            Q_i = q_proj(E_i)                                    # (B, d_proj, L)
            Q_i_t = Q_i.transpose(1, 2)                          # (B, L, d_proj)

            # Attention: (B, L, d_proj) @ (B, d_c, L) -> scores of shape (B, L, L)? no.
            # We want channel-vs-channel affinities. So:
            # scores = Q_i^T @ K_c  (over batch): (B, d_proj, L) @ (B, L, d_c) = (B, d_proj, d_c)
            scores = torch.matmul(Q_i, Kc_t) / math.sqrt(self.d_c)   # (B, d_proj, d_c)
            attn = F.softmax(scores, dim=-1)                          # softmax over d_c
            # O_chan = attn @ V_c: (B, d_proj, d_c) @ (B, d_c, L) = (B, d_proj, L)
            O_chan = torch.matmul(attn, Vc)                           # (B, d_proj, L)

            # Project back to C_i (Eq. 5's O_chan has C_i channels)
            O_chan_full = out_proj(
                # broadcast O_chan up to d_c channels for the projection layer
                # simpler: pad O_chan with zeros to d_c, then project
                F.pad(O_chan, (0, 0, 0, self.d_c - self.d_proj))
            )                                                          # (B, C_i, L)

            # Residual add + LayerNorm over channel dim (Eq. 5)
            X_i = O_chan_full + skips[i].mean(dim=-1, keepdim=True).expand(-1, -1, self.L)
            X_i = ln(X_i.transpose(1, 2)).transpose(1, 2)              # (B, C_i, L)
            cca_outs.append(X_i)

        # -- TCA: standard self-attention over time L --
        Qt = self.tca_q(E_c).transpose(1, 2)                          # (B, L, d_c)
        Kt = self.tca_k(E_c).transpose(1, 2)                          # (B, L, d_c)
        Vt = self.tca_v(E_c).transpose(1, 2)                          # (B, L, d_c)

        B_, L_, d_c_ = Qt.shape
        # Multi-head split
        Qt = Qt.view(B_, L_, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, L, d_k)
        Kt = Kt.view(B_, L_, self.n_heads, self.d_k).transpose(1, 2)
        Vt = Vt.view(B_, L_, self.n_heads, self.d_k).transpose(1, 2)

        sim_t = torch.matmul(Qt, Kt.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_t = F.softmax(sim_t, dim=-1)                             # (B, h, L, L)
        O_time = torch.matmul(attn_t, Vt)                             # (B, h, L, d_k)
        O_time = O_time.transpose(1, 2).contiguous().view(B_, L_, d_c_)
        O_time = O_time.transpose(1, 2)                               # (B, d_c, L)

        tca_outs = []
        for i, (out_proj, ln) in enumerate(zip(self.tca_out, self.tca_ln)):
            Y_i = out_proj(O_time)                                     # (B, C_i, L)
            Y_i = F.gelu(ln(Y_i.transpose(1, 2)).transpose(1, 2))     # LN + GeLU
            tca_outs.append(Y_i)

        # -- Combine (Eq. 9) --
        F_outs = [xi + yi for xi, yi in zip(cca_outs, tca_outs)]      # each (B, C_i, L)

        # -- Interpolate back to original skip lengths and residually add --
        # PyTorch 1.10.x has no bf16 kernel for linear interpolate; use fp32.
        result = []
        for f, orig_skip in zip(F_outs, skips):
            orig_dtype = f.dtype
            with torch.cuda.amp.autocast(enabled=False):
                f_up = F.interpolate(f.float(), size=orig_skip.shape[-1],
                                     mode="linear", align_corners=False)
            f_up = f_up.to(orig_dtype)
            result.append(f_up + orig_skip)                            # residual with original
        return result


# =========================================================================
# THE FULL DCA-SLEEP MODEL
# =========================================================================
class DCASleepInner(nn.Module):
    """
    The DCA-Sleep model as specified in the paper, processing ONE 20-min
    window (40 epochs at 128 Hz) at a time.

    Input:  (B, 1, WINDOW_SMP_PAD) = (B, 1, 163,840)
    Output: (B, 4, 40)
    """
    def __init__(self, num_classes: int = 4,
                 n_epochs_per_window: int = WINDOW_EPOCHS):
        super().__init__()
        self.num_classes = num_classes
        self.n_epochs_per_window = n_epochs_per_window

        # Encoder — 12 blocks, downsample /2 each = 4096x
        # Input length WINDOW_SMP_PAD = 40 * 4096, so we end at 40 tokens
        self.encoder_blocks = nn.ModuleList()
        in_c = 1
        for c in CHANNELS:
            self.encoder_blocks.append(EncoderBlock(in_c, c))
            in_c = c

        # DCA in the middle (operates on all 12 skip connections)
        self.dca = DCAModule(CHANNELS, L=n_epochs_per_window)

        # Decoder — 12 blocks mirroring encoder
        # Deepest first, mirror channel progression back down
        self.decoder_blocks = nn.ModuleList()
        rev_channels = list(reversed(CHANNELS))   # [256]*8 + [128, 64, 32, 16]
        for i in range(12):
            if i == 0:
                # First decoder block: input = deepest bottleneck (from encoder final pool)
                in_ch = CHANNELS[-1]            # 256
                out_ch = rev_channels[i]        # 256
            else:
                # Subsequent: input = prev decoder output CONCAT with corresponding skip
                in_ch = rev_channels[i - 1] + rev_channels[i]
                out_ch = rev_channels[i]
            self.decoder_blocks.append(DecoderBlock(in_ch, out_ch))

        # Post-processing (per paper §2.1 end)
        self.post_pool = nn.AdaptiveMaxPool1d(n_epochs_per_window)     # collapse to 40 tokens
        self.post_conv1 = nn.Conv1d(CHANNELS[0], CHANNELS[0], kernel_size=1)  # 1x1
        self.classifier = nn.Conv1d(CHANNELS[0], num_classes, kernel_size=1)  # 1x1 -> 4 classes

    def forward(self, x):
        """
        x: (B, 1, WINDOW_SMP_PAD)
        Returns: (B, num_classes, WINDOW_EPOCHS) logits
        """
        # -- Encoder path, collect skips (pre-pool) --
        skips = []
        h = x
        for block in self.encoder_blocks:
            h, skip = block(h)
            skips.append(skip)
        # After all 12 blocks: h has shape (B, 256, WINDOW_EPOCHS) i.e. (B, 256, 40)

        # -- DCA enhances the 12 skip connections --
        enhanced_skips = self.dca(skips)

        # -- Decoder path with skip concatenation --
        # Start from deepest (which is the bottleneck output h)
        d = h                                                # (B, 256, 40)
        d = self.decoder_blocks[0](d)                        # (B, 256, 80) after upsample

        for i in range(1, 12):
            skip = enhanced_skips[-(i + 1)]                  # mirror index (11-i in original order)
            # Align temporal length if needed
            # PyTorch 1.10.x has no bf16 kernel for linear interpolate; use fp32.
            if d.shape[-1] != skip.shape[-1]:
                orig_dtype = d.dtype
                with torch.cuda.amp.autocast(enabled=False):
                    d = F.interpolate(d.float(), size=skip.shape[-1],
                                      mode="linear", align_corners=False)
                d = d.to(orig_dtype)
            d = torch.cat([d, skip], dim=1)                  # concat along channel dim
            d = self.decoder_blocks[i](d)

        # -- Post-processing --
        d = self.post_pool(d)                                # (B, 16, 40)
        d = F.elu(self.post_conv1(d))                        # 1x1 conv + activation
        logits = self.classifier(d)                          # (B, 4, 40)
        return logits


# =========================================================================
# WRAPPER FOR OUR HARNESS
# =========================================================================
class DCASleep(nn.Module):
    """
    Harness-facing wrapper. Handles:
      1. Resampling from Fs=125 Hz (our data) to Fs=128 Hz (paper's rate)
      2. Splitting the full recording into non-overlapping 40-epoch windows
      3. Padding each window from 153,600 to 163,840 samples
      4. Running DCASleepInner per window
      5. Concatenating outputs, trimming to actual epoch count
      6. Returning {"main": (B, L, 4), "latent": None}
    """

    def __init__(self, num_classes: int = 4, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.inner = DCASleepInner(num_classes=num_classes,
                                    n_epochs_per_window=WINDOW_EPOCHS)

    # ---------------------------------------------------------------------
    def _prepare_recording(self, raw):
        """
        Turn (B, L, 1, T=3750) at Fs=125 Hz into a list of 20-min windows
        each padded to WINDOW_SMP_PAD = 163,840 samples at Fs=128 Hz.

        Returns:
            windows: (B * n_windows, 1, WINDOW_SMP_PAD) — batched for the inner model
            n_windows: number of windows per recording
            L: number of real epochs in the recording (input L)
        """
        B, L, C, T = raw.shape
        assert C == 1, "Expecting single-channel PPG"

        # Flatten epoch axis into time -> (B, 1, L * T)
        x = raw.permute(0, 2, 1, 3).reshape(B, 1, L * T)       # (B, 1, N_125)

        # Resample from 125 Hz to 128 Hz per-epoch to match paper's Fs
        # target per-epoch samples at 128 Hz: DCA_EPOCH_SMP = 3840
        target_len = L * DCA_EPOCH_SMP
        x = F.interpolate(x, size=target_len,
                          mode="linear", align_corners=False)   # (B, 1, L * 3840)

        # Split into 40-epoch windows (padded at the end if needed)
        n_windows = (L + WINDOW_EPOCHS - 1) // WINDOW_EPOCHS    # ceil
        padded_L = n_windows * WINDOW_EPOCHS
        pad_epochs = padded_L - L
        if pad_epochs > 0:
            pad_samples = pad_epochs * DCA_EPOCH_SMP
            x = F.pad(x, (0, pad_samples), mode="constant", value=0.0)
        # Now x has shape (B, 1, padded_L * 3840)

        # Split into windows: reshape to (B, n_windows, 1, WINDOW_SMP_RAW)
        x = x.view(B, 1, n_windows, WINDOW_SMP_RAW).permute(0, 2, 1, 3)  # (B, n_windows, 1, WINDOW_SMP_RAW)
        x = x.reshape(B * n_windows, 1, WINDOW_SMP_RAW)         # batch-flatten windows

        # Pad each window from WINDOW_SMP_RAW=153,600 to WINDOW_SMP_PAD=163,840
        # so 12 downsamples of 2x land exactly at 40 tokens
        pad_within = WINDOW_SMP_PAD - WINDOW_SMP_RAW           # 10,240
        x = F.pad(x, (0, pad_within), mode="constant", value=0.0)
        # x now (B * n_windows, 1, WINDOW_SMP_PAD)

        return x, n_windows, L

    # ---------------------------------------------------------------------
    def forward(self, raw, epoch_idx=None, mask=None):
        """
        Args:
            raw:       (B, L, 1, T) at Fs=125 Hz
            epoch_idx: unused
            mask:      unused
        Returns:
            {"main": (B, L, num_classes), "latent": None}
        """
        B = raw.shape[0]
        L = raw.shape[1]

        windows, n_windows, real_L = self._prepare_recording(raw)

        # Run inner model on all windows batched together
        y = self.inner(windows)                                 # (B*n_windows, 4, 40)

        # Reshape back to (B, n_windows * 40, 4)
        y = y.permute(0, 2, 1)                                  # (B*n_windows, 40, 4)
        y = y.view(B, n_windows, WINDOW_EPOCHS, self.num_classes)
        y = y.reshape(B, n_windows * WINDOW_EPOCHS, self.num_classes)

        # Trim to original L
        y = y[:, :real_L, :]

        return {"main": y, "latent": None}

    # ---------------------------------------------------------------------
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =========================================================================
# Smoke test — replicates the harness's tensor shapes
# =========================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    model = DCASleep(num_classes=4).cuda()

    # Simulate one recording of ~7 hours: L=840 epochs
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

    # Batch test
    B, L = 2, 720
    raw = torch.randn(B, L, C_in, T).cuda()
    with torch.no_grad():
        out = model(raw)
    print(f"\nBatch test: raw {tuple(raw.shape)} -> main {tuple(out['main'].shape)}")