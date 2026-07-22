# =========================================================================
# archive_baseline_results.py
#
# Reads every completed fold's metrics.json from disk and writes a
# permanent snapshot of the baseline benchmark to:
#     /data2/Akbar1/PPG_Stages/benchmark_results/_paper_results/
#
# Output files:
#   per_fold_metrics.csv      — every fold, every model, every cohort
#   summary_by_model.csv      — mean ± std per model per cohort
#   pairwise_stats.json       — V8 vs each baseline: deltas + Wilcoxon + MWU
#   SUMMARY.md                — paper-ready human-readable summary
#   manifest.json             — what was loaded, when, fold counts
#
# Safe to re-run: regenerates all files in place.
# =========================================================================

import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from scipy.stats import wilcoxon, mannwhitneyu
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    print("WARNING: scipy not available — statistical tests skipped")

# ----- configuration -----
ROOT = Path("/data2/Akbar1/PPG_Stages/benchmark_results")
OUT_DIR = ROOT / "_paper_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All models we care about. Reported results use seed 42 x 5 folds for every
# model; v8 also has seeds 1337 and 2024 on disk, unused in the paper.
MAIN_MODELS    = ["v8", "deepsleepnet", "sleeppgnet", "insightsleepnet", "wang_dualstream", "dca_sleep"]
ABLATION_MODELS = ["v8", "enc_dual_same", "enc_single_card",
                   "plain_transformer", "v8_full_with_ultradian"]
ALL_MODELS = sorted(set(MAIN_MODELS) | set(ABLATION_MODELS))
COHORTS = ["mesa", "cfs"]
CLASSES = ["Wake", "Light", "Deep", "REM"]
PRIMARY_METRICS = ["macro_f1", "kappa", "acc", "ECE"]
PER_CLASS = CLASSES


# =========================================================================
# 1. Load every completed fold
# =========================================================================
def load_fold(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)
    return m


all_rows = []
manifest = {}
for cohort in COHORTS:
    manifest[cohort] = {}
    for model in ALL_MODELS:
        model_dir = ROOT / cohort / model
        if not model_dir.exists():
            manifest[cohort][model] = 0
            continue
        n_done = 0
        for fold_dir in sorted(model_dir.glob("seed*_fold*")):
            done_marker = fold_dir / "DONE"
            metrics_path = fold_dir / "metrics.json"
            if not done_marker.exists() or not metrics_path.exists():
                continue
            n_done += 1
            m = load_fold(metrics_path)
            parts = fold_dir.name.replace("seed", "").split("_fold")
            seed = int(parts[0])
            fold = int(parts[1])
            all_rows.append({
                "cohort":   cohort,
                "model":    model,
                "seed":     seed,
                "fold":     fold,
                "dir":      fold_dir.name,
                "n_epochs": m.get("n_epochs"),
                "macro_f1": m["macro_f1"],
                "kappa":    m["kappa"],
                "acc":      m["acc"],
                "ECE":      m["ECE"],
                "Wake":     m["f1_per_class"]["Wake"],
                "Light":    m["f1_per_class"]["Light"],
                "Deep":     m["f1_per_class"]["Deep"],
                "REM":      m["f1_per_class"]["REM"],
            })
        manifest[cohort][model] = n_done

print(f"Loaded {len(all_rows)} completed folds across "
      f"{len([m for c in manifest.values() for m in c.values() if m > 0])} "
      f"(cohort, model) combinations.")


# =========================================================================
# 2. Write per_fold_metrics.csv (every fold, every model)
# =========================================================================
per_fold_path = OUT_DIR / "per_fold_metrics.csv"
fieldnames = ["cohort", "model", "seed", "fold", "dir", "n_epochs",
              "macro_f1", "kappa", "acc", "ECE",
              "Wake", "Light", "Deep", "REM"]
with open(per_fold_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in sorted(all_rows, key=lambda x: (x["cohort"], x["model"], x["seed"], x["fold"])):
        w.writerow(r)
print(f"WROTE  {per_fold_path}  ({len(all_rows)} rows)")


# =========================================================================
# 3. Compute and write summary_by_model.csv (mean ± std per cohort × model)
# =========================================================================
def filter_rows(cohort, model):
    return [r for r in all_rows if r["cohort"] == cohort and r["model"] == model]


def stats(rows, key):
    vals = np.array([r[key] for r in rows])
    return float(vals.mean()), float(vals.std())


summary_rows = []
for cohort in COHORTS:
    for model in ALL_MODELS:
        rows = filter_rows(cohort, model)
        if not rows:
            continue
        row = {"cohort": cohort, "model": model, "n_runs": len(rows),
               "n_seeds": len(set(r["seed"] for r in rows))}
        for k in PRIMARY_METRICS + PER_CLASS:
            mean, std = stats(rows, k)
            row[f"{k}_mean"] = mean
            row[f"{k}_std"]  = std
        summary_rows.append(row)

summary_path = OUT_DIR / "summary_by_model.csv"
if summary_rows:
    summary_fields = (["cohort", "model", "n_runs", "n_seeds"] +
                      [f"{k}_mean" for k in PRIMARY_METRICS + PER_CLASS] +
                      [f"{k}_std"  for k in PRIMARY_METRICS + PER_CLASS])
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"WROTE  {summary_path}  ({len(summary_rows)} rows)")


# =========================================================================
# 4. Paired statistical tests — V8 vs each baseline
# =========================================================================
pairwise = {}
for cohort in COHORTS:
    pairwise[cohort] = {}
    v8_rows = filter_rows(cohort, "v8")
    if not v8_rows:
        continue
    v8_by_fold_seed42 = {r["fold"]: r for r in v8_rows if r["seed"] == 42}
    v8_all = v8_rows                                  # all 15 runs (3 seeds × 5 folds)

    for baseline in ["deepsleepnet", "sleeppgnet", "insightsleepnet", "wang_dualstream", "dca_sleep"]:
        b_rows = filter_rows(cohort, baseline)
        if not b_rows:
            continue
        b_by_fold = {r["fold"]: r for r in b_rows if r["seed"] == 42}
        common = sorted(set(v8_by_fold_seed42) & set(b_by_fold))

        comp = {"baseline": baseline, "n_v8_runs": len(v8_all),
                "n_baseline_runs": len(b_rows),
                "n_paired_folds_seed42": len(common)}

        for metric in ["macro_f1", "kappa", "acc", "Deep", "REM", "Wake", "Light"]:
            v8_mean = np.mean([r[metric] for r in v8_all])
            b_mean  = np.mean([r[metric] for r in b_rows])
            delta_mean = float(v8_mean - b_mean)

            # paired Wilcoxon on seed42 fold-matched pairs (n=5)
            paired_diffs = [v8_by_fold_seed42[fld][metric] - b_by_fold[fld][metric]
                            for fld in common]
            all_v8_wins = bool(all(d > 0 for d in paired_diffs))
            n_v8_wins   = int(sum(1 for d in paired_diffs if d > 0))

            wilcoxon_p = None
            if HAVE_SCIPY and len(paired_diffs) >= 5:
                try:
                    _, wilcoxon_p = wilcoxon(paired_diffs)
                    wilcoxon_p = float(wilcoxon_p)
                except ValueError:
                    wilcoxon_p = None

            # Mann–Whitney U on the full 15 V8 vs 5 baseline (unpaired)
            mwu_p = None
            if HAVE_SCIPY:
                try:
                    _, mwu_p = mannwhitneyu(
                        [r[metric] for r in v8_all],
                        [r[metric] for r in b_rows],
                        alternative="greater")
                    mwu_p = float(mwu_p)
                except Exception:
                    mwu_p = None

            comp[metric] = {
                "v8_mean":              float(v8_mean),
                "baseline_mean":        float(b_mean),
                "delta_mean":           delta_mean,
                "paired_diffs_seed42":  [float(d) for d in paired_diffs],
                "paired_n_v8_wins":     n_v8_wins,
                "paired_all_positive":  all_v8_wins,
                "wilcoxon_p_paired":    wilcoxon_p,
                "mannwhitney_p_greater": mwu_p,
            }
        pairwise[cohort][baseline] = comp

pairwise_path = OUT_DIR / "pairwise_stats.json"
with open(pairwise_path, "w") as f:
    json.dump(pairwise, f, indent=2)
print(f"WROTE  {pairwise_path}")


# =========================================================================
# 5. Manifest (audit trail)
# =========================================================================
manifest_path = OUT_DIR / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump({
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "result_root":  str(ROOT),
        "completed_fold_counts": manifest,
        "total_folds_loaded": len(all_rows),
    }, f, indent=2)
print(f"WROTE  {manifest_path}")


# =========================================================================
# 6. Human-readable paper-ready summary
# =========================================================================
md_lines = []
md_lines.append("# PPG Sleep-Staging Benchmark — Locked Results Snapshot")
md_lines.append("")
md_lines.append(f"_Generated: {datetime.utcnow().isoformat()}Z_")
md_lines.append("")
md_lines.append("## Fold completion counts")
md_lines.append("")
md_lines.append("| Model | MESA folds | CFS folds |")
md_lines.append("|---|---:|---:|")
for model in ALL_MODELS:
    md_lines.append(f"| `{model}` | {manifest['mesa'].get(model, 0)} | "
                    f"{manifest['cfs'].get(model, 0)} |")
md_lines.append("")

for cohort in COHORTS:
    md_lines.append(f"## {cohort.upper()} — primary comparison table")
    md_lines.append("")
    md_lines.append("| Model | n | macro-F1 | κ | Deep F1 | REM F1 | acc |")
    md_lines.append("|---|---:|---|---|---|---|---|")
    for model in MAIN_MODELS:
        rows = filter_rows(cohort, model)
        if not rows:
            continue
        def cell(k):
            m, s = stats(rows, k)
            return f"{m:.4f} ± {s:.4f}"
        md_lines.append(f"| `{model}` | {len(rows)} | "
                        f"{cell('macro_f1')} | {cell('kappa')} | "
                        f"{cell('Deep')} | {cell('REM')} | {cell('acc')} |")
    md_lines.append("")

    # ablation block
    md_lines.append(f"### {cohort.upper()} — V8 ablation")
    md_lines.append("")
    md_lines.append("| Config | n | macro-F1 | κ | Deep F1 | REM F1 |")
    md_lines.append("|---|---:|---|---|---|---|")
    for model in ABLATION_MODELS:
        rows = filter_rows(cohort, model)
        if not rows:
            continue
        def cell(k):
            m, s = stats(rows, k)
            return f"{m:.4f} ± {s:.4f}"
        md_lines.append(f"| `{model}` | {len(rows)} | "
                        f"{cell('macro_f1')} | {cell('kappa')} | "
                        f"{cell('Deep')} | {cell('REM')} |")
    md_lines.append("")

    # paired tests
    md_lines.append(f"### {cohort.upper()} — V8 vs baselines (paired stats)")
    md_lines.append("")
    md_lines.append("| Comparison | Δκ (mean) | folds V8 wins | Wilcoxon p (n=5 paired) | MWU p (15 vs 5, greater) |")
    md_lines.append("|---|---|---|---|---|")
    for baseline, comp in pairwise.get(cohort, {}).items():
        k = comp["kappa"]
        wins = comp["kappa"]["paired_n_v8_wins"]
        wp = f"{k['wilcoxon_p_paired']:.4f}" if k['wilcoxon_p_paired'] is not None else "n/a"
        mp = f"{k['mannwhitney_p_greater']:.4f}" if k['mannwhitney_p_greater'] is not None else "n/a"
        md_lines.append(f"| V8 vs `{baseline}` | "
                        f"{k['delta_mean']:+.4f} | "
                        f"{wins} / {comp['n_paired_folds_seed42']} | "
                        f"{wp} | {mp} |")
    md_lines.append("")

md_path = OUT_DIR / "SUMMARY.md"
with open(md_path, "w") as f:
    f.write("\n".join(md_lines))
print(f"WROTE  {md_path}")


# =========================================================================
# 7. Final console echo
# =========================================================================
print("\n" + "=" * 70)
print(f"All results archived under: {OUT_DIR}")
print("=" * 70)
print("Files created:")
for p in sorted(OUT_DIR.iterdir()):
    size_kb = p.stat().st_size / 1024
    print(f"  {p.name:<32}  {size_kb:6.1f} KB")