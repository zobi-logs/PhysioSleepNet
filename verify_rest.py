# =========================================================================
# verify_rest.py - reproducibility check for the statistical comparisons.
#
# Recomputes the paired Wilcoxon signed-rank tests reported in
# Section IV-A: PhysioSleepNet against each of the five baselines, on
# per-subject Cohen's kappa and on Deep-sleep F1, for both cohorts.
# Prints the mean paired difference, the fold-win count and the p-value.
#
#   python verify_rest.py
#
# Paper claims, reproduced at submission: statistically superior on kappa
# in 8 of 10 cohort-by-baseline comparisons, and on Deep F1 in 9 of 10.
# With five folds the Wilcoxon test cannot return a value below 0.043;
# the scipy small-sample warning is expected and is discussed in the paper.
# =========================================================================
import json, numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
R = Path("benchmark_results")

def per_fold(coh, m, metric):
    out=[]
    for f in range(5):
        d = R/coh/m/f"seed42_fold{f}"
        if metric=="kappa": out.append(json.load(open(d/"metrics.json"))["kappa"])
        else:
            p=np.load(d/"predictions.npz",allow_pickle=True)
            out.append(f1_score(p["y_true"],p["y_pred"],average=None,labels=[0,1,2,3])[2])
    return np.array(out)

BASE=["deepsleepnet","sleeppgnet","insightsleepnet","wang_dualstream","dca_sleep"]
for metric in ["kappa","deepf1"]:
    for coh in ["mesa","cfs"]:
        sup=0
        print(f"\n-- {coh.upper()} {metric} --")
        a=per_fold(coh,"v8",metric)
        for m in BASE:
            b=per_fold(coh,m,metric); d=a-b; p=wilcoxon(a,b).pvalue
            sup += p<0.05
            print(f"  {m:18s} d={d.mean():+.4f}  wins {int((d>0).sum())}/5  p={p:.3f}")
        print(f"  -> significant in {sup}/5")
