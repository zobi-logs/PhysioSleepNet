# =========================================================================
# encoder_ablation.py
#
# Ablates the ONE remaining untested physiological claim in the paper:
# that the dual-path encoder, with a cardiac path at unit dilation and a
# respiratory path at expanding dilation, is a useful inductive bias.
#
# Sec. II-C says the respiratory path "is therefore an architectural
# inductive bias toward the autonomic content in PPG." That has never
# been ablated. Since the bottleneck and the ultradian PE are both now
# excluded as sources of the MESA win, this is the leading candidate.
#
# ------------------------------------------------------------- VARIANTS
#   enc_dual_diff     2 paths, dilations {1,1,1} and {2,4,4}   <- current v8
#                     channels (96,160,192) each.  REFACTOR CONTROL:
#                     must reproduce v8's kappa or the refactor is broken.
#
#   enc_dual_same     2 paths, BOTH dilations {1,1,1}
#                     channels (96,160,192) each.
#                     *** THE DECISIVE COMPARISON ***
#                     Identical parameter count and identical topology to
#                     enc_dual_diff. The ONLY difference is whether the
#                     two paths see different temporal scales. If this
#                     ties enc_dual_diff, the "respiratory path" is just
#                     extra capacity and the physiological story is wrong.
#
#   enc_single_card   1 path, dilations {1,1,1}, channels (144,232,288)
#   enc_single_resp   1 path, dilations {2,4,4}, channels (144,232,288)
#                     Widened to match the dual-path parameter count so a
#                     drop cannot be blamed on lost capacity. Answers the
#                     separate question: does having TWO branches help at
#                     all, versus one branch of the same size?
#
# ------------------------------------------------------------- WHY BOTH
# enc_dual_same isolates DILATION DIVERSITY (capacity held fixed).
# enc_single_*  isolates BRANCH COUNT      (capacity held fixed).
# Running only one of them leaves the other explanation open.
#
# ----------------------------------------------------------------- USAGE
# 1. Put this file next to models.py.
# 2. In models.py, add near the other imports:
#
#        from encoder_ablation import (PhysioSleepNetEncAblation,
#                                      ENCODER_ABLATION_REGISTRY)
#
#    and after MODEL_REGISTRY is defined:
#
#        MODEL_REGISTRY.update(ENCODER_ABLATION_REGISTRY)
#
#    That is the whole integration. has_latent() still returns False for
#    these keys because the class is not literally PhysioSleepNet -- see
#    the note at the bottom of this file for the one-line fix if you want
#    latents dumped for the encoder variants too.
#
# 3. Self-test (no data needed):
#        python encoder_ablation.py
#
# 4. Train:
#        python harness.py --model enc_dual_same --cohort mesa \
#            --seeds 42 --folds 0 1 2 3 4
# =========================================================================

import torch
import torch.nn as nn

from models import PhysioSleepNet, ResConv1D


# =========================================================================
# Configurable multi-path encoder
# =========================================================================
class ConfigurablePulseEncoder(nn.Module):
    """
    Generalization of MultiScalePulseEncoder to an arbitrary number of
    parallel convolutional paths with per-path dilation and channel
    schedules.

    With paths = [ dict(dilations=(1,1,1), channels=(96,160,192)),
                   dict(dilations=(2,4,4), channels=(96,160,192)) ]
    this is structurally identical to the encoder in models.py: same stem,
    same kernel schedule (7,5,3), same strides (2,2,2), same fuse/down/proj,
    same CLS + learned positional embedding + 2-layer intra-epoch
    transformer, same final LayerNorm and CLS readout.

    Input : (N, 1, 3750)   one 30 s epoch of raw PPG,  N = B*L
    Output: (N, d_model)   one epoch embedding
    """

    KERNELS = (7, 5, 3)
    STRIDES = (2, 2, 2)

    def __init__(self, d_model=384, dropout=0.1, paths=None,
                 stem_channels=48, fuse_channels=256, max_tokens=70):
        super().__init__()
        if paths is None:
            paths = [dict(dilations=(1, 1, 1), channels=(96, 160, 192)),
                     dict(dilations=(2, 4, 4), channels=(96, 160, 192))]
        self.path_cfg = paths

        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, 11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(stem_channels), nn.GELU(),
        )                                                    # 3750 -> 1875

        self.paths = nn.ModuleList()
        for cfg in paths:
            dil, ch = cfg["dilations"], cfg["channels"]
            assert len(dil) == len(ch) == 3, "each path needs 3 blocks"
            blocks, c_in = [], stem_channels
            for k, s, d, c_out in zip(self.KERNELS, self.STRIDES, dil, ch):
                blocks.append(ResConv1D(c_in, c_out, k=k, s=s,
                                        dilation=d, drop=dropout))
                c_in = c_out
            self.paths.append(nn.Sequential(*blocks))        # 1875 -> 235

        cat_ch = sum(cfg["channels"][-1] for cfg in paths)
        self.fuse = ResConv1D(cat_ch, fuse_channels, k=3, s=2, drop=dropout)
        self.down = ResConv1D(fuse_channels, fuse_channels, k=3, s=2, drop=dropout)
        self.proj = nn.Conv1d(fuse_channels, d_model, 1)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.intra_tf = nn.TransformerEncoder(enc, num_layers=2)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):                                    # (N, 1, 3750)
        z = self.stem(x)
        outs = [p(z) for p in self.paths]
        m = min(o.shape[-1] for o in outs)                   # length align
        outs = [o[..., :m] for o in outs]
        z = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        z = self.down(self.fuse(z))
        z = self.proj(z).transpose(1, 2)                     # (N, T, d)
        N, T, D = z.shape
        assert T + 1 <= self.pos.shape[1], (
            f"{T + 1} tokens exceeds pos capacity {self.pos.shape[1]}")
        z = torch.cat([self.cls.expand(N, -1, -1), z], dim=1)
        z = z + self.pos[:, :(T + 1), :]
        z = self.intra_tf(z)
        return self.norm(z[:, 0, :])                         # CLS -> (N, d)


# =========================================================================
# Path configurations
# =========================================================================
CARD = (1, 1, 1)          # cardiac: unit dilation, fine pulse morphology
RESP = (2, 4, 4)          # respiratory: expanding dilation, RSA band
CH_DUAL = (96, 160, 192)  # per-path widths in the current encoder
CH_WIDE = (144, 232, 288) # widened single path, ~parameter-matched to dual

ENCODER_VARIANTS = {
    "dual_diff": [dict(dilations=CARD, channels=CH_DUAL),
                  dict(dilations=RESP, channels=CH_DUAL)],
    "dual_same": [dict(dilations=CARD, channels=CH_DUAL),
                  dict(dilations=CARD, channels=CH_DUAL)],
    "single_card": [dict(dilations=CARD, channels=CH_WIDE)],
    "single_resp": [dict(dilations=RESP, channels=CH_WIDE)],
}


# =========================================================================
# Model wrapper
# =========================================================================
class PhysioSleepNetEncAblation(PhysioSleepNet):
    """
    PhysioSleepNet with a swapped pulse encoder. Everything downstream --
    the 12 TFBlockLG backbone, ln_out, the autonomic bottleneck, the loss
    contract -- is inherited unchanged from models.py, so the only thing
    varying across these registry keys is the encoder.

    Defaults match the v8 headline config (ultradian off, bottleneck on).
    """

    def __init__(self, num_classes=4, encoder_variant="dual_diff",
                 d_model=384, dropout=0.1, **kw):
        super().__init__(num_classes=num_classes, d_model=d_model,
                         dropout=dropout, **kw)
        if encoder_variant not in ENCODER_VARIANTS:
            raise KeyError(f"Unknown encoder_variant '{encoder_variant}'. "
                           f"Available: {list(ENCODER_VARIANTS)}")
        self.encoder_variant = encoder_variant
        self.pulse_encoder = ConfigurablePulseEncoder(
            d_model=d_model, dropout=dropout,
            paths=ENCODER_VARIANTS[encoder_variant],
        )


ENCODER_ABLATION_REGISTRY = {
    # refactor control -- must reproduce v8
    "enc_dual_diff": (PhysioSleepNetEncAblation, dict(
        encoder_variant="dual_diff", use_ultradian=False, use_bottleneck=True)),
    # DECISIVE: same params, same topology, only dilation diversity removed
    "enc_dual_same": (PhysioSleepNetEncAblation, dict(
        encoder_variant="dual_same", use_ultradian=False, use_bottleneck=True)),
    # branch-count ablations, parameter-matched
    "enc_single_card": (PhysioSleepNetEncAblation, dict(
        encoder_variant="single_card", use_ultradian=False, use_bottleneck=True)),
    "enc_single_resp": (PhysioSleepNetEncAblation, dict(
        encoder_variant="single_resp", use_ultradian=False, use_bottleneck=True)),
}


# =========================================================================
# Self-test
# =========================================================================
def _count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _self_test():
    from models import MultiScalePulseEncoder

    print("=" * 72)
    print("  ENCODER PARAMETER MATCHING")
    print("=" * 72)
    ref = MultiScalePulseEncoder(d_model=384, dropout=0.1)
    n_ref = _count(ref)
    print(f"  {'models.py MultiScalePulseEncoder':<34s} {n_ref:>12,}   (reference)")
    for name, paths in ENCODER_VARIANTS.items():
        enc = ConfigurablePulseEncoder(d_model=384, dropout=0.1, paths=paths)
        n = _count(enc)
        print(f"  {'enc_' + name:<34s} {n:>12,}   "
              f"{100.0 * n / n_ref:6.2f}% of reference")

    print("\n" + "=" * 72)
    print("  STRUCTURAL EQUIVALENCE  (dual_diff vs models.py encoder)")
    print("=" * 72)
    dd = ConfigurablePulseEncoder(d_model=384, dropout=0.1,
                                  paths=ENCODER_VARIANTS["dual_diff"])
    a = {k: tuple(v.shape) for k, v in ref.state_dict().items()}
    b = {k: tuple(v.shape) for k, v in dd.state_dict().items()}
    same_n = len(a) == len(b)
    shapes_a = sorted(a.values())
    shapes_b = sorted(b.values())
    print(f"  parameter tensors : {len(a)} vs {len(b)}   "
          f"{'MATCH' if same_n else 'MISMATCH'}")
    print(f"  shape multiset    : "
          f"{'MATCH' if shapes_a == shapes_b else 'MISMATCH'}")
    print(f"  total params      : {n_ref:,} vs {_count(dd):,}   "
          f"{'MATCH' if n_ref == _count(dd) else 'MISMATCH'}")

    print("\n" + "=" * 72)
    print("  FORWARD SHAPES  (B=2, L=4, T=3750)")
    print("=" * 72)
    B, L, T = 2, 4, 3750
    raw = torch.randn(B, L, 1, T)
    eix = torch.arange(L).unsqueeze(0).repeat(B, 1)
    msk = torch.ones(B, L, dtype=torch.bool)
    for key, (cls, kw) in ENCODER_ABLATION_REGISTRY.items():
        m = cls(**{**kw, "num_classes": 4}).eval()
        with torch.no_grad():
            out = m(raw, eix, msk)
        lat = out["latent"]
        ok = (tuple(out["main"].shape) == (B, L, 4)
              and lat is not None and tuple(lat.shape) == (B, L, 32))
        print(f"  {key:<18s} main={tuple(out['main'].shape)} "
              f"latent={tuple(lat.shape) if lat is not None else None} "
              f"params={_count(m) / 1e6:.2f}M  {'OK' if ok else 'FAIL'}")

    print("\n" + "=" * 72)
    print("  TOTAL MODEL PARAMS vs v8")
    print("=" * 72)
    from models import build_model
    n_v8 = _count(build_model("v8", num_classes=4))
    print(f"  {'v8 (models.py)':<20s} {n_v8 / 1e6:7.3f} M   (reference)")
    for key, (cls, kw) in ENCODER_ABLATION_REGISTRY.items():
        n = _count(cls(**{**kw, "num_classes": 4}))
        print(f"  {key:<20s} {n / 1e6:7.3f} M   "
              f"{100.0 * n / n_v8:6.2f}%  (delta {n - n_v8:+,})")
    print()


if __name__ == "__main__":
    _self_test()