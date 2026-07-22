import json
from pathlib import Path
import numpy as np

results_root = Path("/data2/Akbar1/PPG_Stages/benchmark_results")
configs = ["v8", "enc_dual_same", "enc_single_card",
           "plain_transformer", "v8_full_with_ultradian"]

def load_metrics(cohort, model):
    """Load every completed fold's metrics for a (cohort, model)."""
    rows = []
    for d in sorted((results_root / cohort / model).glob("seed42_fold*")):
        if not (d / "DONE").exists():
            continue
        with open(d / "metrics.json") as f:
            m = json.load(f)
        rows.append({
            "macroF1": m["macro_f1"],
            "kappa":   m["kappa"],
            "acc":     m["acc"],
            "Deep":    m["f1_per_class"]["Deep"],
            "REM":     m["f1_per_class"]["REM"],
            "Wake":    m["f1_per_class"]["Wake"],
            "Light":   m["f1_per_class"]["Light"],
            "ECE":     m["ECE"],
        })
    return rows

for cohort in ["mesa", "cfs"]:
    print(f"\n{'='*98}")
    print(f"V8 ABLATION — {cohort.upper()} test (seed 42, mean ± std across 5 folds)")
    print(f"{'='*98}")
    print(f"{'config':<22} {'n':>3} {'macroF1':>14} {'kappa':>14} {'Deep':>14} {'REM':>14} {'acc':>14}")
    print("-" * 98)
    baseline = None
    for c in configs:
        rows = load_metrics(cohort, c)
        if not rows:
            print(f"{c:<22} -- no results --")
            continue
        n = len(rows)
        arr = lambda k: np.array([r[k] for r in rows])
        def cell(k): return f"{arr(k).mean():.4f}±{arr(k).std(ddof=1):.4f}"
        print(f"{c:<22} {n:>3} {cell('macroF1'):>14} {cell('kappa'):>14} "
              f"{cell('Deep'):>14} {cell('REM'):>14} {cell('acc'):>14}")
        if c == "v8":
            baseline = {k: arr(k).mean() for k in ["macroF1","kappa","Deep","REM"]}

    # deltas vs V8 (the main model)
    if baseline is not None:
        print()
        print(f"DELTAS vs v8 (mean only, sign tells direction):")
        for c in configs:
            if c == "v8":
                continue
            rows = load_metrics(cohort, c)
            if not rows:
                continue
            arr = lambda k: np.array([r[k] for r in rows])
            def d(k): return arr(k).mean() - baseline[k]
            print(f"  {c:<22}  dMacroF1={d('macroF1'):+.4f}  dKappa={d('kappa'):+.4f}  "
                  f"dDeep={d('Deep'):+.4f}  dREM={d('REM'):+.4f}")