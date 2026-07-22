import json
import numpy as np
from pathlib import Path

results_root = Path("/data2/Akbar1/PPG_Stages/benchmark_results")

for cohort in ["mesa", "cfs"]:
    print(f"\n========== {cohort.upper()} V8 — per-fold metrics ==========")
    rows = []
    for d in sorted((results_root / cohort / "v8").glob("seed*_fold*")):
        if not (d / "DONE").exists():
            continue
        with open(d / "metrics.json") as f:
            m = json.load(f)
        rows.append({
            "fold": d.name,
            "n_epochs": m["n_epochs"],
            "macroF1": m["macro_f1"],
            "kappa":   m["kappa"],
            "acc":     m["acc"],
            "Deep":    m["f1_per_class"]["Deep"],
            "REM":     m["f1_per_class"]["REM"],
            "Wake":    m["f1_per_class"]["Wake"],
            "Light":   m["f1_per_class"]["Light"],
            "ECE":     m["ECE"],
        })
    if not rows:
        print("  (no completed folds)")
        continue
    # print table
    print(f"  {'fold':<22} {'mF1':>6} {'kappa':>6} {'Deep':>6} {'REM':>6} {'acc':>6} {'ECE':>6} {'#eps':>8}")
    for r in rows:
        print(f"  {r['fold']:<22} {r['macroF1']:6.3f} {r['kappa']:6.3f} "
              f"{r['Deep']:6.3f} {r['REM']:6.3f} {r['acc']:6.3f} {r['ECE']:6.3f} {r['n_epochs']:8d}")
    # summary
    arr = lambda k: np.array([r[k] for r in rows])
    print(f"\n  Mean ± std across {len(rows)} folds:")
    for k in ["macroF1", "kappa", "Deep", "REM", "Wake", "Light", "acc"]:
        v = arr(k)
        print(f"    {k:<8} {v.mean():.4f} ± {v.std():.4f}  (min {v.min():.4f}, max {v.max():.4f})")