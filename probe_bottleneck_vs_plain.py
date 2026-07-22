#!/usr/bin/env python
# =========================================================================
# probe_bottleneck_vs_plain.py
#
# Drop this in /data2/Akbar1/PPG_Stages/ next to harness.py and
# linear_probe.py. It imports both, so it inherits your frozen folds,
# your eval preprocessing, and your HRV feature extraction. Nothing is
# reimplemented -- the R2 values it produces are directly comparable to
# Figure 3.
#
# ---------------------------------------------------------------- QUESTION
# Table IV shows the bottleneck is accuracy-neutral. The paper's remaining
# justification is interpretability. Does the 32-dim bottleneck actually
# CONCENTRATE autonomic information into a linearly-accessible form, or
# would any transformer representation decode HRV just as well?
#
# ------------------------------------------------------------- THE CONFOUND
# v8's latent is 32-dim; a backbone output is 384-dim. A ridge probe with
# 384 inputs has ~12x the capacity of one with 32. Comparing them as-is
# measures dimensionality, not architecture. Everything below reduces both
# to k PCA components and probes at matched k, swept over a range.
#
# ------------------------------------------------------------------- ARMS
#   v8_latent      (32)   v8's bottleneck output z  [also on disk in
#                         latents.npz; recomputed here and cross-checked]
#   v8_backbone    (384)  v8's ln_out, i.e. PRE-bottleneck
#   plain_backbone (384)  plain_transformer's ln_out
#
# The DECISIVE comparison is v8_latent vs v8_backbone@32PC. Same network,
# same weights up to ln_out, same training run -- the only difference is
# whether the bottleneck was applied. No cross-run confound. plain_backbone
# is the cross-model sanity check.
#
# ------------------------------------------ PRE-REGISTERED DECISION RULE
# Commit this to git BEFORE running. Primary targets are HR_std (SDNN) and
# RMSSD, chosen because they are the two the paper already reports as
# strongly decoded (R2 0.49 / 0.44).
#
#   The concentration claim SURVIVES iff, on MESA, for BOTH SDNN and RMSSD:
#       R2(v8_latent) - R2(v8_backbone @ k=32) > pooled fold-std
#
#   Otherwise: TIE -> the bottleneck is a practical device, not a
#   mechanism, and Sec. II-E / IV-B must say so.
#
# Do not revise this rule after seeing output. Do not switch primary
# targets to HF_power because it happened to come out favourably.
#
# ------------------------------------------------------------------ USAGE
#   python probe_bottleneck_vs_plain.py --list
#   python probe_bottleneck_vs_plain.py --cohort mesa --fold 0 --seed 42
#   for f in 0 1 2 3 4; do
#       python probe_bottleneck_vs_plain.py --cohort mesa --fold $f --seed 42
#   done
#   python probe_bottleneck_vs_plain.py --aggregate --cohort mesa
#
# Inference only. Read-only w.r.t. training code: representations are
# pulled with a forward hook, so the no-bottleneck path is NOT modified
# (which would silently re-enable L_smooth for those variants).
# =========================================================================

import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path("/data2/Akbar1/PPG_Stages")
OUT_DIR = PROJECT_ROOT / "probe_bottleneck_results"

K_SWEEP = [2, 4, 8, 16, 32, 64, 128, 384]
ALPHA_GRID = np.logspace(-3, 5, 17)
PRIMARY_TARGETS = ["HR_std", "RMSSD"]      # pre-registered
ARM_NAMES = ["v8_latent", "v8_backbone", "plain_backbone"]


# ============================================================ module import
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_project_modules():
    """Import harness.py and linear_probe.py and locate what we need."""
    H = _load_module(PROJECT_ROOT / "harness.py", "proj_harness")
    LP = _load_module(PROJECT_ROOT / "linear_probe.py", "proj_linear_probe")

    need_h = ["RESULTS_ROOT", "COHORT_PATHS", "get_or_make_folds",
              "split_dfs_for_fold"]
    for n in need_h:
        if not hasattr(H, n):
            raise RuntimeError(f"harness.py has no '{n}' -- did it move?")

    if not hasattr(LP, "_features_from_epoch"):
        raise RuntimeError("linear_probe.py has no '_features_from_epoch'")
    targets = getattr(LP, "TARGETS",
                      ["HR_mean", "HR_std", "RMSSD", "HF_power", "resp_rate"])

    # locate the eval Dataset class + collate fn without hardcoding names
    import torch.utils.data as tud
    ds_cls, collate = None, None
    for _, obj in inspect.getmembers(H):
        if inspect.isclass(obj) and issubclass(obj, tud.Dataset) and obj is not tud.Dataset:
            ds_cls = obj
        if inspect.isfunction(obj) and obj.__name__.lower().startswith("collate"):
            collate = obj
    if ds_cls is None:
        raise RuntimeError("Could not find a Dataset subclass in harness.py")
    print(f"  harness dataset : {ds_cls.__name__}")
    print(f"  harness collate : {collate.__name__ if collate else '(none)'}")
    print(f"  probe targets   : {targets}")
    return H, LP, ds_cls, collate, targets


# ============================================================ checkpoint I/O
def resolve_fold_dir(H, cohort, model, seed, fold):
    """Find the run dir, checking the live tree then the _LOCKED_ archives."""
    root = Path(H.RESULTS_ROOT)
    cands = [
        root / cohort / model / f"seed{seed}_fold{fold}",
        root / f"_LOCKED_{model}_{cohort}" / f"seed{seed}_fold{fold}",
        root / f"_LOCKED_{model}_main_{cohort}" / f"seed{seed}_fold{fold}",
        root / f"_LOCKED_{model}_{cohort}_main" / f"seed{seed}_fold{fold}",
    ]
    for c in cands:
        if (c / "checkpoint.pt").exists():
            return c
    raise FileNotFoundError(
        f"No checkpoint.pt for model={model} cohort={cohort} "
        f"seed={seed} fold={fold}. Tried:\n  " +
        "\n  ".join(str(c) for c in cands) +
        "\nRun with --list to see what exists."
    )


def list_available(H):
    root = Path(H.RESULTS_ROOT)
    print(f"\nResults root: {root}\n")
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == "folds":
            continue
        runs = sorted(p.parent.name for p in d.rglob("checkpoint.pt"))
        if runs:
            print(f"  {d.name}/  ->  {len(runs)} runs: {runs[:3]}"
                  f"{' ...' if len(runs) > 3 else ''}")
        else:
            for sub in sorted(d.iterdir()):
                if sub.is_dir():
                    r = sorted(p.parent.name for p in sub.rglob("checkpoint.pt"))
                    if r:
                        print(f"  {d.name}/{sub.name}/  ->  {len(r)} runs: "
                              f"{r[:3]}{' ...' if len(r) > 3 else ''}")
    fp = root / "folds"
    if fp.exists():
        print(f"\nFrozen folds: {[f.name for f in sorted(fp.glob('*.json'))]}")


def build_and_load(models_mod, registry_key, ckpt_dir, num_classes, device):
    import torch
    model = models_mod.build_model(registry_key, num_classes=num_classes)
    ckpt = torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu")
    state = None
    for k in ("model_state", "state_dict", "model_state_dict"):
        if isinstance(ckpt, dict) and k in ckpt:
            state = ckpt[k]
            break
    if state is None:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"    missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"    e.g. missing: {missing[:4]}")
    if len(missing) > 20:
        raise RuntimeError(
            f"{len(missing)} missing keys loading {ckpt_dir} as "
            f"'{registry_key}'. Wrong registry key, or a notebook-trained "
            f"checkpoint (ultradian.proj.* vs ultradian_proj.*). Refusing "
            f"to probe a half-initialised model."
        )
    return model.to(device).eval()


class LnOutTap:
    """Forward hook on model.ln_out. Read-only, no graph modification."""

    def __init__(self, model):
        self.buf = None
        self.handle = model.ln_out.register_forward_hook(
            lambda m, i, o: setattr(self, "buf", o.detach()))

    def close(self):
        self.handle.remove()


# ============================================================== extraction
def extract(H, LP, models_mod, ds_cls, collate, targets, cohort, fold, seed,
            device, num_classes, max_epochs_per_subject, rng_seed):
    import torch
    from torch.utils.data import DataLoader

    fold_info = H.get_or_make_folds(cohort)
    manifest_df = pd.read_csv(H.COHORT_PATHS[cohort]["manifest"])
    _, _, te_df = H.split_dfs_for_fold(manifest_df, fold_info, fold)
    print(f"  test fold {fold}: {len(te_df)} recordings, "
          f"{te_df['subject_id'].nunique()} subjects")

    v8_dir = resolve_fold_dir(H, cohort, "v8", seed, fold)
    pl_dir = resolve_fold_dir(H, cohort, "plain_transformer", seed, fold)
    print(f"  v8    <- {v8_dir}")
    print(f"  plain <- {pl_dir}")

    print("  loading models ...")
    m_v8 = build_and_load(models_mod, "v8", v8_dir, num_classes, device)
    m_pl = build_and_load(models_mod, "plain_transformer", pl_dir,
                          num_classes, device)
    tap_v8, tap_pl = LnOutTap(m_v8), LnOutTap(m_pl)

    # harness's own eval dataset -> identical preprocessing to training eval
    try:
        ds = ds_cls(te_df, mode="eval")
    except TypeError:
        ds = ds_cls(te_df, "eval")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                        collate_fn=collate) if collate else \
        DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    amp_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if amp_ok else torch.float32
    rng = np.random.RandomState(rng_seed)

    acc = {a: [] for a in ARM_NAMES}
    acc_t, acc_s, acc_y = [], [], []
    paths = te_df["npz_path"].tolist()
    subjs = te_df["subject_id"].astype(str).tolist()

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc="  extract")):
            raw, y, eix, mask = [b.to(device) for b in batch[:4]]
            with torch.cuda.amp.autocast(enabled=amp_ok, dtype=amp_dtype):
                out_v8 = m_v8(raw, eix, mask)
                _ = m_pl(raw, eix, mask)

            if out_v8["latent"] is None:
                raise RuntimeError(
                    "v8 returned latent=None -> that checkpoint was trained "
                    "with use_bottleneck=False. Wrong directory.")

            m0 = mask[0].cpu().numpy().astype(bool)
            rep = {
                "v8_latent":      out_v8["latent"].float()[0].cpu().numpy()[m0],
                "v8_backbone":    tap_v8.buf.float()[0].cpu().numpy()[m0],
                "plain_backbone": tap_pl.buf.float()[0].cpu().numpy()[m0],
            }
            yv = y[0].cpu().numpy()[m0]
            E = rep["v8_latent"].shape[0]

            # HRV targets from the RAW npz epochs, using linear_probe's
            # extractor -> comparable to Figure 3
            try:
                arr = np.load(paths[bi], allow_pickle=True)
                xr, yr = arr["x"].astype(np.float32), arr["y"].astype(np.int64)
                xr = xr[yr >= 0]
            except Exception as e:
                print(f"    skip {paths[bi]}: {e}")
                continue
            if xr.shape[0] != E:
                print(f"    length mismatch {paths[bi]}: raw {xr.shape[0]} "
                      f"vs model {E} -- skipping")
                continue

            idx = np.arange(E)
            if E > max_epochs_per_subject:
                idx = np.sort(rng.choice(E, max_epochs_per_subject, False))

            tg = np.full((len(idx), len(targets)), np.nan, dtype=np.float32)
            for j, i in enumerate(idx):
                try:
                    f = LP._features_from_epoch(xr[i])
                    if isinstance(f, dict):
                        tg[j] = [f.get(k, np.nan) for k in targets]
                    else:
                        tg[j] = np.asarray(f, dtype=np.float32)[:len(targets)]
                except Exception:
                    pass

            ok = np.isfinite(tg).any(axis=1)
            if not ok.any():
                continue
            keep = idx[ok]
            for a in ARM_NAMES:
                acc[a].append(rep[a][keep])
            acc_t.append(tg[ok])
            acc_y.append(yv[keep])
            acc_s.append(np.array([subjs[bi]] * int(ok.sum()), dtype=object))

    tap_v8.close(); tap_pl.close()
    if not acc_t:
        raise RuntimeError("No usable epochs extracted.")

    data = {
        "reps": {a: np.concatenate(acc[a], 0).astype(np.float32) for a in ARM_NAMES},
        "targets": np.concatenate(acc_t, 0).astype(np.float32),
        "subjects": np.concatenate(acc_s, 0),
        "stages": np.concatenate(acc_y, 0).astype(np.int64),
        "target_names": targets,
    }

    # sanity: our recomputed latent should match the harness's saved one
    lp = v8_dir / "latents.npz"
    if lp.exists():
        try:
            z_saved = np.load(lp)["z"]
            print(f"  [check] saved latents {z_saved.shape} vs recomputed "
                  f"{data['reps']['v8_latent'].shape} (subsampled -- shapes "
                  f"differ by design; std {z_saved.std():.3f} vs "
                  f"{data['reps']['v8_latent'].std():.3f})")
        except Exception:
            pass
    return data


# ================================================================= probing
def _r2(yt, yp):
    ssr = float(np.sum((yt - yp) ** 2))
    sst = float(np.sum((yt - np.mean(yt)) ** 2))
    return 1.0 - ssr / max(sst, 1e-12)


def probe_one(X, y, subjects, k, seed=42):
    """Subject-disjoint 60/20/20. PCA fit on train only. Alpha tuned on val
    per (arm, k, target). R2 on held-out test subjects."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(subjects)
    rng.shuffle(uniq)
    n = len(uniq)
    if n < 5:
        return np.nan
    tr = set(uniq[:int(0.6 * n)])
    va = set(uniq[int(0.6 * n):int(0.8 * n)])
    m_tr = np.array([s in tr for s in subjects])
    m_va = np.array([s in va for s in subjects])
    m_te = ~(m_tr | m_va)
    if m_tr.sum() < 50 or m_va.sum() < 20 or m_te.sum() < 20:
        return np.nan

    k_eff = min(k, X.shape[1], int(m_tr.sum()) - 1)
    if k_eff < 1:
        return np.nan

    xs = StandardScaler().fit(X[m_tr])
    pca = PCA(n_components=k_eff, random_state=seed).fit(xs.transform(X[m_tr]))
    Ztr = pca.transform(xs.transform(X[m_tr]))
    Zva = pca.transform(xs.transform(X[m_va]))
    Zte = pca.transform(xs.transform(X[m_te]))

    ys = StandardScaler().fit(y[m_tr].reshape(-1, 1))
    ytr = ys.transform(y[m_tr].reshape(-1, 1)).ravel()
    yva = ys.transform(y[m_va].reshape(-1, 1)).ravel()
    yte = ys.transform(y[m_te].reshape(-1, 1)).ravel()

    best_a, best_v = ALPHA_GRID[0], -np.inf
    for a in ALPHA_GRID:
        v = _r2(yva, Ridge(alpha=a).fit(Ztr, ytr).predict(Zva))
        if v > best_v:
            best_v, best_a = v, a
    return _r2(yte, Ridge(alpha=best_a).fit(Ztr, ytr).predict(Zte))


def sweep(data, seed=42):
    rows = []
    for ti, tname in enumerate(data["target_names"]):
        y = data["targets"][:, ti]
        ok = np.isfinite(y)
        if ok.sum() < 200:
            print(f"  {tname}: only {ok.sum()} finite -- skipped")
            continue
        for arm in ARM_NAMES:
            X = data["reps"][arm][ok]
            for k in K_SWEEP:
                if k > X.shape[1]:
                    continue
                r2 = probe_one(X, y[ok], data["subjects"][ok], k, seed)
                rows.append(dict(target=tname, arm=arm, k=k, dim=X.shape[1],
                                 r2=r2, n_rows=int(ok.sum())))
                print(f"    {tname:10s} {arm:15s} k={k:4d}  R2={r2:+.4f}")
    return pd.DataFrame(rows)


# =============================================================== aggregate
def aggregate(cohort):
    files = sorted(OUT_DIR.glob(f"sweep_{cohort}_*.csv"))
    if not files:
        print(f"No sweep_{cohort}_*.csv in {OUT_DIR}")
        return
    df = pd.concat([pd.read_csv(f).assign(src=f.stem) for f in files])
    g = df.groupby(["target", "arm", "k"])["r2"].agg(["mean", "std", "count"]).reset_index()
    g.to_csv(OUT_DIR / f"AGGREGATE_{cohort}.csv", index=False)

    print("\n" + "=" * 76)
    print(f"  MATCHED-DIMENSIONALITY VERDICT — {cohort.upper()}  "
          f"(k=32, mean±std over {len(files)} folds)")
    print("=" * 76)
    verdict = {}
    for t in sorted(set(g.target)):
        sub = g[(g.target == t) & (g.k == 32)]
        row = {r["arm"]: (r["mean"], r["std"]) for _, r in sub.iterrows()}
        if "v8_latent" not in row:
            continue
        lm, ls = row["v8_latent"]
        star = " *PRIMARY*" if t in PRIMARY_TARGETS else ""
        print(f"\n  {t}{star}")
        print(f"    v8_latent       R2 = {lm:+.4f} ± {ls:.4f}")
        for other in ["v8_backbone", "plain_backbone"]:
            if other not in row:
                continue
            om, osd = row[other]
            d = lm - om
            pooled = float(np.sqrt(ls ** 2 + osd ** 2)) or 1e-9
            call = ("CONCENTRATES" if d > pooled else
                    "DEGRADES" if d < -pooled else "TIE")
            print(f"    {other:15s} R2 = {om:+.4f} ± {osd:.4f}"
                  f"   Δ={d:+.4f}  -> {call}")
            verdict[f"{t}|{other}"] = dict(delta=float(d), call=call)

    json.dump(verdict, open(OUT_DIR / f"VERDICT_{cohort}.json", "w"), indent=2)

    calls = [verdict[f"{t}|v8_backbone"]["call"]
             for t in PRIMARY_TARGETS if f"{t}|v8_backbone" in verdict]
    print("\n" + "-" * 76)
    print("  PRE-REGISTERED RULE (primary targets vs v8's OWN backbone):")
    if calls and all(c == "CONCENTRATES" for c in calls):
        print("  -> SURVIVES. The bottleneck concentrates autonomic")
        print("     information at matched dimensionality. Report as a")
        print("     quantified architectural contribution.")
    elif calls and any(c == "DEGRADES" for c in calls):
        print("  -> DEGRADES. The bottleneck loses linearly-accessible HRV")
        print("     information vs its own backbone. Report it; do not bury it.")
    else:
        print("  -> TIE. The bottleneck does not concentrate beyond what the")
        print("     backbone already exposes. Reframe Sec. II-E / IV-B: it is")
        print("     a practical device (tractable probing, t-SNE, smoothness")
        print("     penalty) at zero accuracy cost, not a mechanism.")
    print("-" * 76)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = [t for t in ["HR_std", "RMSSD", "HF_power", "HR_mean"]
              if t in set(g.target)]
        fig, ax = plt.subplots(1, len(ts), figsize=(4.2 * len(ts), 3.6),
                               squeeze=False)
        for a, t in zip(ax[0], ts):
            for arm, st in [("v8_latent", "o-"), ("v8_backbone", "s--"),
                            ("plain_backbone", "^:")]:
                s = g[(g.target == t) & (g.arm == arm)].sort_values("k")
                if s.empty:
                    continue
                a.errorbar(s["k"], s["mean"], yerr=s["std"], fmt=st, capsize=3,
                           label=arm, lw=1.4, ms=5)
            a.set_xscale("log", base=2); a.axvline(32, c="grey", lw=.8, ls=":")
            a.set_xlabel("PCA components k"); a.set_ylabel("held-out $R^2$")
            a.set_title(t); a.grid(alpha=.25)
        ax[0][0].legend(fontsize=8)
        fig.suptitle(f"{cohort.upper()}: HRV decodability at matched "
                     f"dimensionality", y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"r2_vs_k_{cohort}.png", dpi=200,
                    bbox_inches="tight")
        print(f"\nFigure -> {OUT_DIR / f'r2_vs_k_{cohort}.png'}")
    except Exception as e:
        print(f"[plot skipped] {e}")


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--cohort", default="mesa", choices=["mesa", "cfs"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--max-epochs-per-subject", type=int, default=200)
    ap.add_argument("--gpu", default="1")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.aggregate:
        aggregate(args.cohort)
        return

    print("Importing project modules ...")
    H, LP, ds_cls, collate, targets = load_project_modules()

    if args.list:
        list_available(H)
        return

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    models_mod = _load_module(PROJECT_ROOT / "models.py", "proj_models")

    data = extract(H, LP, models_mod, ds_cls, collate, targets, args.cohort,
                   args.fold, args.seed, device, args.num_classes,
                   args.max_epochs_per_subject, args.seed)

    print(f"\n  {data['targets'].shape[0]} epochs / "
          f"{len(np.unique(data['subjects']))} subjects")
    for a in ARM_NAMES:
        print(f"    {a:15s} {data['reps'][a].shape}")

    tag = f"{args.cohort}_seed{args.seed}_fold{args.fold}"
    np.savez_compressed(
        OUT_DIR / f"reps_{tag}.npz",
        targets=data["targets"], subjects=data["subjects"].astype(str),
        stages=data["stages"], target_names=np.array(targets),
        **{f"rep_{a}": data["reps"][a].astype(np.float16) for a in ARM_NAMES})

    print("\n  sweeping k ...")
    sweep(data, seed=args.seed).to_csv(OUT_DIR / f"sweep_{tag}.csv", index=False)
    print(f"\n  wrote {OUT_DIR / f'sweep_{tag}.csv'}")
    print(f"  when all folds done: python {Path(__file__).name} "
          f"--aggregate --cohort {args.cohort}")


if __name__ == "__main__":
    main()