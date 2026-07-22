# =========================================================================
# models.py — model registry for the PPG sleep-staging benchmark.
#
# CONTRACT
#   Every model in MODEL_REGISTRY must satisfy:
#       forward(raw, epoch_idx, mask) -> dict
#       raw       : (B, L, 1, 3750)  float
#       epoch_idx : (B, L)           long   (0-indexed within recording)
#       mask      : (B, L)           bool
#       returns   : {"main": (B, L, C) logits,
#                    "latent": (B, L, K) or None   # K=32 for V8 bottleneck}
#
#   Models that don't use epoch_idx (most baselines) ignore the arg. The
#   uniform signature lets the harness call every model identically.
#
#   Models that don't have an interpretable latent return "latent": None.
#   The harness only writes a latents.npz when latent is present.
#
# REGISTRY
#   build_model(name, num_classes) -> nn.Module
#   Add new models by adding an entry to MODEL_REGISTRY.
# =========================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, "/data2/Akbar1/PPG_Stages/baselines")
from wang_dualstream import WangDualStreamBaseline
from dca_sleep import DCASleep
# =========================================================================
# Shared building blocks
# =========================================================================
def _rotate_half(x):
    x1 = x[..., ::2]; x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RoPE(nn.Module):
    """Rotary positional encoding — relative position via complex rotation."""
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.base = base

    def forward(self, x):
        B, L, H, Dh = x.shape
        half = Dh // 2
        freqs = 1.0 / (self.base ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
        t = torch.arange(L, device=x.device, dtype=torch.float32)
        ang = torch.einsum("l,d->ld", t, freqs)
        cos = torch.cos(ang)[None, :, None, :].repeat_interleave(2, -1).to(x.dtype)
        sin = torch.sin(ang)[None, :, None, :].repeat_interleave(2, -1).to(x.dtype)
        return (x * cos) + (_rotate_half(x) * sin)


class DropPath(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if (not self.training) or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x / keep * torch.floor(keep + torch.rand(shape, device=x.device))


def _windows(L, w):
    out, s = [], 0
    while s < L:
        e = min(L, s + w)
        out.append((s, e))
        s = e
    return out


class ResConv1D(nn.Module):
    """Residual 1D conv with optional dilation."""
    def __init__(self, c_in, c_out, k, s=1, dilation=1, drop=0.0):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=s, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(c_out), nn.GELU(), nn.Dropout(drop),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.skip = (nn.Conv1d(c_in, c_out, 1, stride=s, bias=False)
                     if (c_in != c_out or s != 1) else nn.Identity())
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class AttnRoPE_LG(nn.Module):
    """RoPE attention with local-window OR full-global mode."""
    def __init__(self, d_model=384, n_heads=8, dropout=0.1, window=180):
        super().__init__()
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.w = int(window)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.rope = RoPE(self.dh)

    def forward(self, x, mask=None, global_attn=False):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.dh)
        k = k.view(B, L, self.n_heads, self.dh)
        v = v.view(B, L, self.n_heads, self.dh)
        q = self.rope(q); k = self.rope(k)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        if global_attn or self.w >= L:
            sc = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
            sc = sc.float()
            if mask is not None:
                sc = sc.masked_fill(~mask[:, None, None, :], -1e9)
            at = torch.softmax(sc, -1)
            out = self.drop(at).to(v.dtype) @ v
        else:
            out = torch.zeros((B, self.n_heads, L, self.dh), device=x.device, dtype=v.dtype)
            for (s, e) in _windows(L, self.w):
                qs, ks, vs = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e]
                sc = (qs @ ks.transpose(-2, -1)) / math.sqrt(self.dh)
                sc = sc.float()
                if mask is not None:
                    sc = sc.masked_fill(~mask[:, None, None, s:e], -1e9)
                at = torch.softmax(sc, -1)
                out[:, :, s:e] = self.drop(at).to(vs.dtype) @ vs
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


class TFBlockLG(nn.Module):
    def __init__(self, d_model=384, n_heads=8, drop=0.1, dp=0.1, window=180):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = AttnRoPE_LG(d_model, n_heads, drop, window=window)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Dropout(drop), nn.Linear(4 * d_model, d_model),
        )
        self.dp = DropPath(dp)

    def forward(self, x, mask, global_attn=False):
        x = x + self.dp(self.attn(self.ln1(x), mask, global_attn=global_attn))
        x = x + self.dp(self.mlp(self.ln2(x)))
        return x


# =========================================================================
# V8 components
# =========================================================================
class MultiScalePulseEncoder(nn.Module):
    """
    Per-epoch encoder for raw PPG (3750 samples @ 125Hz).
      cardiac path : small kernels for pulse-by-pulse morphology
      respiratory path : dilated convs for ~3-5s breathing modulation
    Both paths fuse; a 2-layer intra-epoch transformer summarizes via a CLS
    token to a single epoch embedding.
    """
    def __init__(self, d_model=384, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 48, 11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(48), nn.GELU(),
        )
        self.card1 = ResConv1D(48, 96,  k=7, s=2, dilation=1, drop=dropout)
        self.card2 = ResConv1D(96, 160, k=5, s=2, dilation=1, drop=dropout)
        self.card3 = ResConv1D(160, 192, k=3, s=2, dilation=1, drop=dropout)
        self.resp1 = ResConv1D(48, 96,  k=7, s=2, dilation=2, drop=dropout)
        self.resp2 = ResConv1D(96, 160, k=5, s=2, dilation=4, drop=dropout)
        self.resp3 = ResConv1D(160, 192, k=3, s=2, dilation=4, drop=dropout)
        self.fuse = ResConv1D(192 * 2, 256, k=3, s=2, drop=dropout)
        self.down = ResConv1D(256, 256, k=3, s=2, drop=dropout)
        self.proj = nn.Conv1d(256, d_model, 1)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.intra_tf = nn.TransformerEncoder(enc, num_layers=2)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, 70, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        z = self.stem(x)
        c = self.card3(self.card2(self.card1(z)))
        r = self.resp3(self.resp2(self.resp1(z)))
        if c.shape[-1] != r.shape[-1]:
            m = min(c.shape[-1], r.shape[-1])
            c = c[..., :m]; r = r[..., :m]
        z = torch.cat([c, r], dim=1)
        z = self.down(self.fuse(z))
        z = self.proj(z).transpose(1, 2)
        BL, Ttok, D = z.shape
        z = torch.cat([self.cls.expand(BL, -1, -1), z], dim=1)
        z = z + self.pos[:, :(Ttok + 1), :]
        z = self.intra_tf(z)
        return self.norm(z[:, 0, :])


class AutonomicBottleneck(nn.Module):
    """
    d_model -> latent_dim "autonomic state" -> linear readout to stage.
    The latent is the interpretability anchor; the harness saves it on
    test passes and the analysis stage linear-probes it against HR/HRV/etc.
    """
    def __init__(self, d_model, latent_dim=32, num_classes=4, dropout=0.1):
        super().__init__()
        self.to_latent = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, latent_dim),
        )
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = self.latent_norm(self.to_latent(x))
        return self.classifier(z), z


# =========================================================================
# V8 — PhysioSleepNet (single-stream raw PPG)
# =========================================================================
class PhysioSleepNet(nn.Module):
    """
    The main model. Bottleneck ON by default = headline V8 config.

    Toggles for ablation:
        use_ultradian   : ultradian positional encoding (block [B])
        use_bottleneck  : autonomic bottleneck (block [D])
    The two main ablation variants set these flags accordingly.
    """
    def __init__(self,
                 num_classes=4, d_model=384, depth=12, n_heads=8,
                 window=180, global_every=3, dropout=0.1,
                 latent_dim=32,
                 use_ultradian=False,           # V8 ablation showed False is better
                 use_bottleneck=True,
                 epochs_per_cycle=180,
                 typical_night=960):
        super().__init__()
        self.use_ultradian = use_ultradian
        self.use_bottleneck = use_bottleneck
        self.global_every = global_every
        self.epochs_per_cycle = epochs_per_cycle
        self.typical_night = typical_night
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.pulse_encoder = MultiScalePulseEncoder(d_model=d_model, dropout=dropout)

        # Ultradian PE — kept as a module even if disabled, for clean state-dict
        if use_ultradian:
            self.ultradian_proj = nn.Sequential(
                nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model),
            )

        self.blocks = nn.ModuleList([
            TFBlockLG(d_model=d_model, n_heads=n_heads, drop=dropout,
                      dp=dropout * (i + 1) / depth, window=window)
            for i in range(depth)
        ])
        self.ln_out = nn.LayerNorm(d_model)

        if use_bottleneck:
            self.bottleneck = AutonomicBottleneck(d_model, latent_dim, num_classes, dropout)
        else:
            self.head = nn.Linear(d_model, num_classes)

    def _ultradian_features(self, epoch_idx):
        t = epoch_idx.float()
        ph1 = 2 * math.pi * t / self.epochs_per_cycle
        ph2 = 2 * math.pi * t / (self.epochs_per_cycle / 2.0)
        return torch.stack([
            torch.sin(ph1), torch.cos(ph1),
            torch.sin(ph2), torch.cos(ph2),
            (t / self.typical_night).clamp(0, 2.0),
        ], dim=-1)

    def forward(self, raw, epoch_idx, mask):
        B, L, C, T = raw.shape
        # [A] per-epoch encoder
        z = self.pulse_encoder(raw.view(B * L, C, T)).view(B, L, -1)
        # [B] ultradian PE (optional)
        if self.use_ultradian:
            feats = self._ultradian_features(epoch_idx)
            z = z + self.ultradian_proj(feats.to(z.dtype))
        # [C] inter-epoch transformer
        for i, blk in enumerate(self.blocks):
            use_global = (self.global_every > 0) and (i % self.global_every == 0)
            z = blk(z, mask, global_attn=use_global)
        z = self.ln_out(z)
        # [D] bottleneck (optional)
        if self.use_bottleneck:
            logits, latent = self.bottleneck(z)
            return {"main": logits, "latent": latent}
        return {"main": self.head(z), "latent": None}



# =========================================================================
# DeepSleepNet — PPG-adapted baseline
#
# Original:  Supratak, Dong, Wu & Guo (2017). "DeepSleepNet: a Model for
#            Automatic Sleep Stage Scoring based on Raw Single-Channel EEG."
#            IEEE TNSRE.
# Adaptation: kernel sizes scaled from Fs=100 (EEG) to Fs=125 (PPG) using
#             the same formulas as the original paper:
#               small-filter kernel  ≈ Fs/2  = 63   (odd for symmetric pad)
#               small-filter stride  ≈ Fs/16 = 8
#               large-filter kernel  ≈ Fs*4  = 501
#               large-filter stride  ≈ Fs/2  = 62
#
# Architecture (per V8 harness contract):
#   forward(raw, epoch_idx, mask) -> {"main": (B,L,C) logits, "latent": None}
#   epoch_idx is accepted but unused (this baseline has no positional prior).
#
# Per-epoch CNN with two parallel branches:
#   small-filter branch -> captures fine temporal structure
#   large-filter branch -> captures slower (frequency-like) structure
#   -> concat -> projection -> 2-layer bidirectional LSTM over the epoch
#   sequence -> shortcut add (CNN repr + BiLSTM out) -> linear classifier
#
# Parameter count: ~5.1 M  (vs ~27.6 M for V8 — appropriate baseline scale)
#
# INSTALL
#   Append this file's contents to your existing models.py, then add ONE
#   line to MODEL_REGISTRY in models.py:
#
#       "deepsleepnet": (DeepSleepNetPPG, dict()),
#
#   No harness changes needed.  has_latent("deepsleepnet") correctly returns
#   False because the existing has_latent() only returns True for
#   PhysioSleepNet — DeepSleepNet will skip the latent dump automatically.
# =========================================================================

import torch
import torch.nn as nn


class _CNNBranchSmall(nn.Module):
    """
    Small-filter branch (temporal features).
    Input : (N, 1, T_EPOCH=3750) raw PPG
    Output: (N, 128, ~14)
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.body = nn.Sequential(
            # large stride conv to compress the signal
            nn.Conv1d(1, 64, kernel_size=63, stride=8, padding=31, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(dropout),
            # three stacked small-kernel convs (same-padding)
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

    def forward(self, x):
        return self.body(x)


class _CNNBranchLarge(nn.Module):
    """
    Large-filter branch (slower / frequency-like features).
    Input : (N, 1, T_EPOCH=3750)
    Output: (N, 128, ~7)
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=501, stride=62, padding=250, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.body(x)


class DeepSleepNetPPG(nn.Module):
    """
    DeepSleepNet adapted to raw PPG at 125 Hz, 30 s epochs.
    Two parallel CNN branches per epoch -> concat -> projection ->
    bidirectional LSTM over the epoch sequence -> shortcut add ->
    linear classifier.

    Required harness signature:
        forward(raw, epoch_idx, mask) -> {"main": (B,L,C), "latent": None}
    epoch_idx is accepted but unused (DeepSleepNet has no positional prior).
    The harness's loss code adds the autonomic-smoothness term only when
    latent is not None, so this baseline trains under CE + focal only —
    consistent with the original paper.

    Server-specific fixes:
      * flatten_parameters is stubbed out to bypass the cuDNN 8.0.5 vs 8.3.2
        version mismatch on this machine. LSTM falls back to native CUDA
        kernels (slightly slower, fully correct).
      * The LSTM forward is wrapped in an autocast(enabled=False) block
        because PyTorch's fused LSTM cell has no bfloat16 kernel in 1.10.x.
        Inputs are cast to fp32 around the LSTM only; the rest of the model
        runs in bf16 as before.
    """

    def __init__(self,
                 num_classes: int = 4,
                 fs: int = 125,
                 lstm_hidden: int = 256,
                 lstm_layers: int = 2,
                 proj_dim: int = 512,
                 dropout: float = 0.5):
        super().__init__()
        assert 2 * lstm_hidden == proj_dim, \
            "Set lstm_hidden so that 2*lstm_hidden == proj_dim (shortcut add)."

        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.epoch_samples = fs * 30      # 3750 at Fs=125

        self.cnn_small = _CNNBranchSmall(dropout=dropout)
        self.cnn_large = _CNNBranchLarge(dropout=dropout)

        # discover concatenated-feature dimension with a dry forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.epoch_samples)
            fs_out = self.cnn_small(dummy)
            fl_out = self.cnn_large(dummy)
            self.flat_small = int(fs_out.shape[1] * fs_out.shape[2])
            self.flat_large = int(fl_out.shape[1] * fl_out.shape[2])
        cat_dim = self.flat_small + self.flat_large

        self.cnn_dropout = nn.Dropout(dropout)
        self.cnn_proj = nn.Linear(cat_dim, proj_dim)

        self.bilstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # Bypass cuDNN init — the system cuDNN version mismatches PyTorch's.
        # LSTM still works via native CUDA kernels; just no cuDNN speedup.
        self.bilstm.flatten_parameters = lambda: None

        self.dropout_after = nn.Dropout(dropout)
        self.classifier = nn.Linear(proj_dim, num_classes)

    def _encode_epochs(self, raw_flat):
        """raw_flat: (B*L, 1, T_EPOCH) -> (B*L, proj_dim)"""
        s = self.cnn_small(raw_flat).flatten(1)   # (B*L, flat_small)
        l = self.cnn_large(raw_flat).flatten(1)   # (B*L, flat_large)
        z = torch.cat([s, l], dim=1)              # (B*L, cat_dim)
        z = self.cnn_dropout(z)
        return self.cnn_proj(z)                   # (B*L, proj_dim)

    def forward(self, raw, epoch_idx, mask):
        # raw: (B, L, 1, T_EPOCH); epoch_idx and mask are accepted but
        # the LSTM runs over the padded sequence — masked positions still
        # produce outputs which the harness's loss masks out.
        B, L, C, T = raw.shape
        cnn_feat = self._encode_epochs(raw.view(B * L, C, T))   # (B*L, proj)
        cnn_feat = cnn_feat.view(B, L, self.proj_dim)           # (B, L, proj)

        # PyTorch's LSTM lacks a bfloat16 kernel here.
        # Run LSTM in fp32, cast back to the surrounding autocast dtype.
        orig_dtype = cnn_feat.dtype
        with torch.cuda.amp.autocast(enabled=False):
            lstm_out, _ = self.bilstm(cnn_feat.float())         # (B, L, proj)
        lstm_out = lstm_out.to(orig_dtype)

        # shortcut add — CNN representation re-injected before the head
        z = lstm_out + cnn_feat
        z = self.dropout_after(z)
        logits = self.classifier(z)                             # (B, L, C)
        return {"main": logits, "latent": None}


# =========================================================================
# MODEL_REGISTRY UPDATE
# Append this line to the MODEL_REGISTRY dict in your models.py:
# =========================================================================
#
#     "deepsleepnet": (DeepSleepNetPPG, dict()),
#
# That's it.  has_latent("deepsleepnet") correctly returns False because
# the existing has_latent() function in models.py only returns True for
# PhysioSleepNet — so the harness will skip latent dumps for this model.




# =========================================================================
# SleepPPG-Net  —  baseline reimplementation
#
# Source paper:
#   Kotzen K, Charlton PH, Salabi S, Amar L, Landesberg A, Behar JA.
#   "SleepPPG-Net: A Deep Learning Algorithm for Robust Sleep Staging
#   From Continuous Photoplethysmography."
#   IEEE J. Biomed. Health Inform., 27(2), pp. 924-933, Feb 2023.
#
# This is a faithful reimplementation from the published architecture
# description (Section II.D.3 and Fig. 2). The authors did not release
# training code at the time of writing; we follow the paper's
# specification exactly where stated.
#
# ARCHITECTURE
#   Input: raw PPG @ whatever Fs (we get 125 Hz from the harness)
#   1. Resample to 34.13 Hz internally (the paper's preprocessing step)
#      -> 1024 samples per 30 s epoch.
#   2. Whole-night ResConv encoder over the CONTINUOUS signal:
#      8 stacked ResConv blocks with channels [16,16,32,32,64,64,128,256].
#      Each ResConv = 3 stacked Conv1d (kernel 3, LeakyReLU) + MaxPool(2)
#      with a residual addition that also pools by 2 and projects channels.
#      Total downsampling: 2^8 = 256x. With 1024 samples per epoch the
#      encoder output has exactly 4 timesteps per epoch.
#   3. Window/reshape: (B, 256, L*4) -> (B, L, 1024)
#      = 4 timesteps * 256 channels collapsed into 1024-dim per epoch.
#   4. Time-distributed Dense(128) projection.
#   5. Two stacked TCN blocks. Each TCN = 6 dilated Conv1d (kernel 7,
#      dilations [1,2,4,8,16,32], 128 channels) with per-layer residual
#      addition and dropout 0.2. Receptive field per block is ~378 epochs;
#      two blocks together span essentially the whole night.
#   6. 1x1 Conv classifier per epoch -> num_classes logits.
#
# ADAPTATIONS DOCUMENTED FOR THE PAPER
#   * Resampling to 34.13 Hz is done inside the model so the harness can
#     feed 125 Hz PPG without changes to the data pipeline.
#   * No SHHS ECG pretraining (the paper shows it does not meaningfully
#     help SleepPPG-Net itself; only the BM-DTS baseline benefits from it).
#   * No transfer-learning step: we evaluate the model trained from scratch
#     under our unified 5-fold protocol. The paper's TL results require a
#     fine-tuning loop with held-out CFS folds, which would not be
#     comparable to our other baselines.
#
# REQUIRED HARNESS SIGNATURE
#   forward(raw, epoch_idx, mask) -> {"main": (B,L,C), "latent": None}
#   epoch_idx and mask accepted but unused (SleepPPG-Net has no positional
#   prior; mask is handled in the harness loss).
#
# INSTALL
#   Paste the contents of this file (everything between `import torch`
#   and the "MODEL_REGISTRY UPDATE" comment) at the bottom of models.py,
#   then add ONE line to MODEL_REGISTRY:
#       "sleeppgnet": (SleepPPGNet, dict()),
#
# Expected param count: ~2.0-2.5 M  (smaller than V8 — paper's design is
# convolution-heavy with a thin TCN, not transformer-style).
# =========================================================================
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# =========================================================================
# ResConv block — encoder building block
# =========================================================================
class _ResConv1D(nn.Module):
    """
    One ResConv block from SleepPPG-Net:
      x -> Conv(k=3) -> LReLU -> Conv(k=3) -> LReLU -> Conv(k=3) -> LReLU
        -> MaxPool(2)
      shortcut: x -> 1x1 Conv (channel projection) -> MaxPool(2)
      output = LReLU(main + shortcut)
    """
 
    def __init__(self, c_in: int, c_out: int, k: int = 3, leaky: float = 0.01):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(c_in,  c_out, k, padding=pad)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad)
        self.conv3 = nn.Conv1d(c_out, c_out, k, padding=pad)
        self.act = nn.LeakyReLU(leaky, inplace=True)
        self.pool = nn.MaxPool1d(2)
        # 1x1 shortcut for channel matching; pooling done in forward
        self.skip_proj = nn.Conv1d(c_in, c_out, 1)
 
    def forward(self, x):
        # main path
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = self.pool(h)
        # shortcut path
        s = self.skip_proj(x)
        s = F.max_pool1d(s, 2)
        return self.act(h + s)
 
 
# =========================================================================
# TCN block — sequence encoder building block
# =========================================================================
class _TCNBlock(nn.Module):
    """
    One TCN block: a stack of dilated Conv1d layers with per-layer
    residual addition and dropout.
 
    Paper figure shows dilations [1, 2, 4, 8, 16, 32], kernel 7, 128
    channels, dropout 0.2. (The paper text says "5 dilated convolutions";
    the figure shows 6. We follow the figure, which is more explicit.)
    """
 
    def __init__(self, channels: int = 128, kernel: int = 7,
                 dilations=(1, 2, 4, 8, 16, 32), dropout: float = 0.2,
                 leaky: float = 0.01):
        super().__init__()
        self.dilated_convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for d in dilations:
            pad = (kernel - 1) * d // 2          # "same" padding given dilation
            self.dilated_convs.append(
                nn.Conv1d(channels, channels, kernel, padding=pad, dilation=d)
            )
            self.dropouts.append(nn.Dropout(dropout))
        self.act = nn.LeakyReLU(leaky, inplace=True)
 
    def forward(self, x):
        for conv, drop in zip(self.dilated_convs, self.dropouts):
            residual = x
            x = self.act(conv(x))
            x = drop(x)
            x = x + residual                     # per-layer residual
        return x
 
 
# =========================================================================
# SleepPPG-Net
# =========================================================================
class SleepPPGNet(nn.Module):
    """
    Faithful reimplementation of Kotzen et al. (2023), IEEE JBHI.
    See file header for full architecture details and adaptation notes.
    """
 
    def __init__(self,
                 num_classes: int = 4,
                 fs_in: int = 125,                          # harness sampling rate
                 fs_target: float = 34.13,                  # paper's internal Fs
                 channels_list=(16, 16, 32, 32, 64, 64, 128, 256),
                 dense_dim: int = 128,
                 tcn_channels: int = 128,
                 tcn_dilations=(1, 2, 4, 8, 16, 32),
                 n_tcn_blocks: int = 2,
                 dropout: float = 0.2,
                 leaky: float = 0.01):
        super().__init__()
        self.fs_in = fs_in
        self.fs_target = fs_target
        # paper choice: 34.13 Hz * 30 s = 1024 samples per epoch (= 2^10).
        self.epoch_samples_target = int(round(fs_target * 30))   # 1024
 
        # ----- 8 ResConv blocks -----
        self.encoder = nn.ModuleList()
        c_prev = 1
        for c in channels_list:
            self.encoder.append(_ResConv1D(c_prev, c, k=3, leaky=leaky))
            c_prev = c
        self.encoder_channels_out = channels_list[-1]            # 256
        self.downsample_factor = 2 ** len(channels_list)         # 2^8 = 256
 
        # After encoder, each epoch has (epoch_samples_target / 256) = 4 timesteps,
        # with `encoder_channels_out` channels. Flatten to one vector per epoch.
        self.timesteps_per_epoch = self.epoch_samples_target // self.downsample_factor
        assert self.epoch_samples_target % self.downsample_factor == 0, \
            "fs_target * 30 must be a multiple of 2^len(channels_list)."
        self.embed_per_epoch = self.timesteps_per_epoch * self.encoder_channels_out  # 1024
 
        # ----- time-distributed Dense projection -----
        self.dense = nn.Linear(self.embed_per_epoch, dense_dim)
        self.dense_act = nn.LeakyReLU(leaky, inplace=True)
 
        # ----- project dense_dim -> tcn_channels if they differ -----
        self.proj_to_tcn = (nn.Identity() if dense_dim == tcn_channels
                            else nn.Conv1d(dense_dim, tcn_channels, 1))
 
        # ----- stacked TCN blocks -----
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(tcn_channels, kernel=7, dilations=tcn_dilations,
                      dropout=dropout, leaky=leaky)
            for _ in range(n_tcn_blocks)
        ])
 
        # ----- 1x1 conv classifier -----
        self.classifier = nn.Conv1d(tcn_channels, num_classes, 1)
 
    def forward(self, raw, epoch_idx, mask):
        """
        raw       : (B, L, 1, T_in)   T_in = fs_in * 30 = 3750 at 125 Hz
        epoch_idx : (B, L)            ignored — no positional prior
        mask      : (B, L)            ignored here; the harness loss masks
                                      padded epochs out of the gradient.
        returns   : {"main": (B, L, num_classes), "latent": None}
        """
        B, L, C, T_in = raw.shape
 
        # ----- (1) concatenate all epochs into a continuous whole-night signal -----
        # (paper processes the whole night as a single time series)
        x = raw.view(B, 1, L * T_in)                                 # (B, 1, L*T_in)
 
        # ----- (2) resample to 34.13 Hz internally (paper's preprocessing) -----
        # PyTorch's linear interpolate lacks a bfloat16 kernel in older versions,
        # so we cast to fp32 around the op and restore the original dtype after.
        orig_dtype = x.dtype
        target_len = L * self.epoch_samples_target
        x = F.interpolate(x.float(), size=target_len,
                          mode="linear", align_corners=False)        # (B, 1, L*1024)
        x = x.to(orig_dtype)
 
        # ----- (3) 8 ResConv blocks over the continuous signal -----
        for block in self.encoder:
            x = block(x)
        # x: (B, 256, L * 4)
        c_out = x.shape[1]
        assert x.shape[2] == L * self.timesteps_per_epoch, \
            f"encoder output length mismatch: got {x.shape[2]}, expected {L * self.timesteps_per_epoch}"
 
        # ----- (4) Window / reshape to per-epoch features -----
        # (B, 256, L*4) -> (B, 256, L, 4) -> (B, L, 256, 4) -> (B, L, 1024)
        x = x.view(B, c_out, L, self.timesteps_per_epoch)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, L, self.embed_per_epoch)                       # (B, L, 1024)
 
        # ----- (5) Dense projection -----
        x = self.dense_act(self.dense(x))                            # (B, L, 128)
 
        # ----- (6) TCN over the epoch sequence -----
        x = x.transpose(1, 2)                                        # (B, 128, L)
        x = self.proj_to_tcn(x)
        for tcn in self.tcn_blocks:
            x = tcn(x)                                               # (B, 128, L)
 
        # ----- (7) classifier -----
        logits = self.classifier(x)                                  # (B, num_classes, L)
        logits = logits.transpose(1, 2)                              # (B, L, num_classes)
        return {"main": logits, "latent": None}

# =========================================================================
# InsightSleepNet  —  baseline reimplementation
#
# Source paper:
#   Nam B, Bark B, Lee J, Kim IY.
#   "InsightSleepNet: the interpretable and uncertainty-aware deep learning
#    network for sleep staging using continuous Photoplethysmography."
#   BMC Medical Informatics and Decision Making (2024) 24:50
#
# This reimplementation follows the architecture description in Section
# "Methods - InsightSleepNet" and Figure 1 of the paper.
#
# ARCHITECTURE
#   Input: raw PPG @ 125 Hz from the harness, resampled internally to
#   34.13 Hz (1024 samples per 30-s epoch).
#
#   1. Local Attention Module
#      - Causal Conv1d with kernel = 7 epochs = 7168 samples (the
#        paper's "3-minute rule" receptive field), with sample-resolution
#        sigmoid attention applied multiplicatively to the input.
#   2. InceptionTime Module
#      - Initial Conv1d (channels=32, kernel=40, stride=20)
#      - 6 stacked Inception blocks with channel sizes [32, 32, 64, 64,
#        128, 256] and bottleneck sizes [8, 16, 16, 16, 32, 32].
#        Each block has bottleneck + 3 parallel kernels (5, 11, 23) +
#        max-pool branch, concatenated and BN+ReLU.
#      - Adaptive average pool to L epochs (one feature vector per epoch).
#      - 1x1 Conv to project to dense_dim = 256.
#   3. Time-Distributed Dense Layer (Conv1d k=1) to dense_dim = 256.
#   4. 5 stacked TCN blocks
#      - Each: causal Conv1d (k=8, dilation=d) → ReLU → Dropout
#              → causal Conv1d (k=8, dilation=d) → ReLU → Dropout
#              → residual addition → ReLU
#      - Dilations [1, 2, 4, 8, 16]; output channels = 64.
#   5. 1x1 Conv classifier → num_classes per epoch.
#
# NOTE ON THE LOCAL ATTENTION
#   The 7168-sample kernel is the most expensive operation in the model.
#   The paper used batch size 2 on an RTX 3090. With our 48 GB A6000 and
#   our 5h training crop (L≈600 epochs) we expect this to fit but you
#   may need to reduce the harness batch size if you see OOM.
#
# WHAT WE DO NOT INCLUDE
#   - Energy-score thresholding: this is a post-hoc inference-time
#     rejection mechanism, not the architecture. The paper's fair
#     comparison numbers are reported BEFORE thresholding:
#         MESA acc=0.842, kappa=0.742
#         CFS  acc=0.806, kappa=0.718
#     Those are the numbers our reimplementation should be compared to.
#   - Transfer learning from MESA to CFS: we train from scratch on each
#     cohort for protocol consistency across all our baselines.
#
# REQUIRED HARNESS SIGNATURE
#   forward(raw, epoch_idx, mask) -> {"main": (B,L,C), "latent": None}
#
# INSTALL
#   Paste this file's contents at the bottom of models.py (everything
#   between `import torch` and the MODEL_REGISTRY UPDATE comment), then
#   add ONE line to MODEL_REGISTRY:
#       "insightsleepnet": (InsightSleepNetPPG, dict()),
# =========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Local Attention Module (TCN-style causal conv with sample-level sigmoid)
# =========================================================================
class _LocalAttentionTCN(nn.Module):
    """
    Causal-convolution local attention from InsightSleepNet.

    The kernel covers the 7 preceding epochs (the paper's "3-minute rule"),
    so each sample's attention value is a learned function of the previous
    ~3.5 minutes of PPG. A sigmoid produces an attention value in [0,1]
    that is multiplied with the input element-wise.
    """

    def __init__(self, epoch_samples: int = 1024, kernel_epochs: int = 7,
                 mid_channels: int = 4):
        super().__init__()
        kernel_size = kernel_epochs * epoch_samples           # 7 * 1024 = 7168
        self.left_pad = kernel_size - 1                       # causal padding
        # large-kernel causal conv on the raw stream
        self.causal_conv = nn.Conv1d(1, mid_channels, kernel_size)
        # 1x1 conv to collapse back to one attention channel
        self.point_conv = nn.Conv1d(mid_channels, 1, 1)

    def forward(self, x):
        # x: (B, 1, T) where T = L * epoch_samples
        x_padded = F.pad(x, (self.left_pad, 0))               # causal pad on left
        h = self.causal_conv(x_padded)                        # (B, mid, T)
        h = self.point_conv(h)                                # (B, 1, T)
        attn = torch.sigmoid(h)                               # (B, 1, T)
        return x * attn                                       # weighted input


# =========================================================================
# InceptionTime Block (bottleneck + 3 parallel kernels + maxpool branch)
# =========================================================================
class _InceptionBlock(nn.Module):
    """
    One InceptionTime block. Input -> bottleneck -> 3 parallel Conv1d (k=5,11,23)
    plus a maxpool->1x1 path; concatenated then BN+ReLU.

    out_channels must be divisible by 4 (the 4 parallel paths each
    produce out_channels/4 channels).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 bottleneck_channels: int = 32, kernels=(5, 11, 23)):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        branch_out = out_channels // 4

        # bottleneck: only used if in_channels > 1
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
            branch_in = bottleneck_channels
        else:
            self.bottleneck = None
            branch_in = in_channels

        # 3 parallel convs with different kernel sizes
        self.conv1 = nn.Conv1d(branch_in, branch_out, kernels[0],
                               padding=kernels[0] // 2, bias=False)
        self.conv2 = nn.Conv1d(branch_in, branch_out, kernels[1],
                               padding=kernels[1] // 2, bias=False)
        self.conv3 = nn.Conv1d(branch_in, branch_out, kernels[2],
                               padding=kernels[2] // 2, bias=False)

        # maxpool branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels, branch_out, 1, bias=False)

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.bottleneck(x) if self.bottleneck is not None else x
        c1 = self.conv1(z)
        c2 = self.conv2(z)
        c3 = self.conv3(z)
        c4 = self.conv4(self.maxpool(x))
        out = torch.cat([c1, c2, c3, c4], dim=1)
        return self.act(self.bn(out))


# =========================================================================
# TCN Block (causal, dilated, with residual)
# =========================================================================
class _TCNBlock(nn.Module):
    """
    InsightSleepNet's temporal block:
      causal Conv1d (k=8, dilation=d) -> ReLU -> Dropout
      causal Conv1d (k=8, dilation=d) -> ReLU -> Dropout
      + residual -> ReLU

    1x1 conv on the residual path if the channel count changes (used in
    the first block which goes from dense_dim -> tcn_channels).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel: int = 8, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel, dilation=dilation)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
        self.residual = (nn.Identity() if in_channels == out_channels
                         else nn.Conv1d(in_channels, out_channels, 1))

    def forward(self, x):
        h = F.pad(x, (self.padding, 0))                       # causal pad
        h = self.act(self.conv1(h))
        h = self.drop1(h)
        h = F.pad(h, (self.padding, 0))
        h = self.act(self.conv2(h))
        h = self.drop2(h)
        return self.act(h + self.residual(x))


# =========================================================================
# InsightSleepNet (full model)
# =========================================================================
class InsightSleepNetPPG(nn.Module):
    """
    Reimplementation of Nam et al. (2024), BMC Medical Informatics.
    See file header for the full architecture rationale and adaptations.
    """

    def __init__(self,
                 num_classes: int = 4,
                 fs_in: int = 125,                            # harness Fs
                 fs_target: float = 34.13,                    # paper's internal Fs
                 init_conv_channels: int = 32,
                 init_conv_kernel: int = 40,
                 init_conv_stride: int = 20,
                 inception_channels=(32, 32, 64, 64, 128, 256),
                 inception_bottlenecks=(8, 16, 16, 16, 32, 32),
                 dense_dim: int = 256,
                 tcn_channels: int = 64,
                 tcn_kernel: int = 8,
                 tcn_dilations=(1, 2, 4, 8, 16),
                 dropout: float = 0.2,
                 local_attn_kernel_epochs: int = 7,
                 local_attn_mid_channels: int = 4):
        super().__init__()
        self.fs_in = fs_in
        self.fs_target = fs_target
        self.epoch_samples_target = int(round(fs_target * 30))   # 1024

        # ----- (1) local attention -----
        self.local_attn = _LocalAttentionTCN(
            epoch_samples=self.epoch_samples_target,
            kernel_epochs=local_attn_kernel_epochs,
            mid_channels=local_attn_mid_channels,
        )

        # ----- (2) initial stride-20 conv -----
        self.init_conv = nn.Conv1d(1, init_conv_channels, init_conv_kernel,
                                    stride=init_conv_stride,
                                    padding=init_conv_kernel // 2)
        self.init_act = nn.ReLU(inplace=True)

        # ----- (3) 6 inception blocks -----
        self.inception_blocks = nn.ModuleList()
        in_ch = init_conv_channels
        for out_ch, bot in zip(inception_channels, inception_bottlenecks):
            self.inception_blocks.append(
                _InceptionBlock(in_ch, out_ch, bottleneck_channels=bot)
            )
            in_ch = out_ch
        self.inception_out_channels = inception_channels[-1]   # 256

        # 1x1 conv after adaptive pool (paper's "Conv 1D" at the
        # bottom of the InceptionTime module box)
        self.after_pool_conv = nn.Conv1d(self.inception_out_channels, dense_dim, 1)

        # ----- (4) time-distributed dense layer -----
        # 1x1 conv across time is equivalent to a time-distributed dense
        self.td_dense = nn.Conv1d(dense_dim, dense_dim, 1)
        self.td_act = nn.ReLU(inplace=True)

        # ----- (5) 5 stacked TCN blocks -----
        self.tcn_blocks = nn.ModuleList()
        in_ch = dense_dim
        for d in tcn_dilations:
            self.tcn_blocks.append(
                _TCNBlock(in_ch, tcn_channels, kernel=tcn_kernel,
                          dilation=d, dropout=dropout)
            )
            in_ch = tcn_channels

        # ----- (6) 1x1 conv classifier -----
        self.classifier = nn.Conv1d(tcn_channels, num_classes, 1)

    def forward(self, raw, epoch_idx, mask):
        """
        raw       : (B, L, 1, T_in)   T_in = fs_in * 30 = 3750 at 125 Hz
        epoch_idx : (B, L)            unused
        mask      : (B, L)            unused here; handled in harness loss
        returns   : {"main": (B, L, num_classes), "latent": None}
        """
        B, L, C, T_in = raw.shape

        # ----- concatenate epochs into whole-night signal -----
        x = raw.view(B, 1, L * T_in)                          # (B, 1, L*3750)

        # ----- resample to 34.13 Hz internally (fp32-wrap for bf16 safety) -----
        orig_dtype = x.dtype
        target_len = L * self.epoch_samples_target
        x = F.interpolate(x.float(), size=target_len,
                          mode="linear", align_corners=False)  # (B, 1, L*1024)
        x = x.to(orig_dtype)

        # ----- local attention -----
        x = self.local_attn(x)                                # (B, 1, L*1024)

        # ----- initial stride-20 conv -----
        x = self.init_act(self.init_conv(x))                  # (B, 32, ~L*51)

        # ----- 6 inception blocks (length-preserving) -----
        for block in self.inception_blocks:
            x = block(x)                                      # (B, 256, ~L*51)

        # ----- adaptive avg pool to L epochs -----
        x = F.adaptive_avg_pool1d(x, L)                       # (B, 256, L)
        x = self.after_pool_conv(x)                           # (B, dense_dim, L)

        # ----- time-distributed dense -----
        x = self.td_act(self.td_dense(x))                     # (B, dense_dim, L)

        # ----- 5 TCN blocks -----
        for tcn in self.tcn_blocks:
            x = tcn(x)                                        # (B, tcn_channels, L)

        # ----- classifier -----
        logits = self.classifier(x)                           # (B, num_classes, L)
        logits = logits.transpose(1, 2)                       # (B, L, num_classes)
        return {"main": logits, "latent": None}


# =========================================================================
# MODEL_REGISTRY UPDATE — append this line to MODEL_REGISTRY in models.py:
# =========================================================================
#
#     "insightsleepnet": (InsightSleepNetPPG, dict()),
#
from baselines.wang_dualstream import WangDualStreamBaseline

# =========================================================================
# MODEL REGISTRY
# =========================================================================
# Each entry: name -> (class, kwargs).  Add baselines in subsequent steps.
MODEL_REGISTRY = {
    # --- V8 main model + ablation variants ---
    "v8": (PhysioSleepNet, dict(
        use_ultradian=False, use_bottleneck=True,
    )),
    "v8_full_with_ultradian": (PhysioSleepNet, dict(
        use_ultradian=True, use_bottleneck=True,
    )),
    "plain_transformer": (PhysioSleepNet, dict(
        use_ultradian=False, use_bottleneck=False,
    )),


    "deepsleepnet": (DeepSleepNetPPG, dict()),
    "sleeppgnet": (SleepPPGNet, dict()),
    #"insightsleepnet": (InsightSleepNetPPG, dict()),
    "insightsleepnet": (InsightSleepNetPPG, dict(local_attn_kernel_epochs=3, local_attn_mid_channels=2)),
    "wang_dualstream": (WangDualStreamBaseline, dict()),
    "dca_sleep": (DCASleep, dict()),

    # --- TODO: baselines added in subsequent steps ---
    # "deepsleepnet":  (DeepSleepNet,  {...}),
    # "sleeppgnet":    (SleepPPGNet,   {...}),
    # "sleeppgnet2":   (SleepPPGNet2,  {...}),
    # "usleep_ppg":    (USleepPPG,     {...}),
}


def build_model(name: str, num_classes: int = 4, **override) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    cls, kwargs = MODEL_REGISTRY[name]
    kwargs = dict(kwargs)
    kwargs["num_classes"] = num_classes
    kwargs.update(override)
    return cls(**kwargs)


def has_latent(name: str) -> bool:
    """Quick check whether a model produces a latent worth saving."""
    cls, kwargs = MODEL_REGISTRY[name]
    if cls is PhysioSleepNet:
        return kwargs.get("use_bottleneck", True)
    return False

# --- encoder ablation variants -------------------------------------------
# MUST stay at end of file: encoder_ablation imports PhysioSleepNet and
# ResConv1D from this module, so it can only be imported once both exist.
from encoder_ablation import PhysioSleepNetEncAblation, ENCODER_ABLATION_REGISTRY
MODEL_REGISTRY.update(ENCODER_ABLATION_REGISTRY)
