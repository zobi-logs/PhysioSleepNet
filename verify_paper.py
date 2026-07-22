# =========================================================================
# verify_paper.py - reproducibility check for the published tables.
#
# Recomputes every value in Table III (24 cells), Table IV (5 means + 5
# standard deviations) and Section IV-D (10 linear-probe R^2 values)
# directly from the per-fold metrics.json and predictions.npz files, and
# diffs each against the number printed in the paper. Expected values are
# hardcoded below so the comparison is explicit.
#
#   python verify_paper.py
#
# Exit state is reported as "N matched, M mismatched". At the time of
# submission: 68 matched, 0 mismatched.
# =========================================================================
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

R = Path("benchmark_results")
ok = bad = 0

def chk(label, got, exp, tol=0.0006):
    global ok, bad
    if got is None:
        print(f"  MISSING  {label}"); bad += 1; return
    flag = "ok " if abs(got-exp) <= tol else "FAIL"
    if flag == "ok ": ok += 1
    else: bad += 1
    print(f"  {flag} {label:38s} paper={exp:+.3f}  computed={got:+.4f}")

def folds(coh, m):
    ks, fs, ds, rs = [], [], [], []
    for f in range(5):
        d = R/coh/m/f"seed42_fold{f}"
        if not (d/"metrics.json").exists(): return None
        j = json.load(open(d/"metrics.json")); ks.append(j["kappa"]); fs.append(j["macro_f1"])
        p = np.load(d/"predictions.npz", allow_pickle=True)
        c = f1_score(p["y_true"], p["y_pred"], average=None, labels=[0,1,2,3])
        ds.append(c[2]); rs.append(c[3])
    g = lambda v: (np.mean(v), np.std(v, ddof=1))
    return dict(kappa=g(ks), macro=g(fs), deep=g(ds), rem=g(rs))

# ---- Table III --------------------------------------------------------
T3 = {
 "mesa": {"deepsleepnet":(.677,.714,.464,.730), "sleeppgnet":(.704,.739,.487,.782),
          "insightsleepnet":(.676,.722,.489,.747), "wang_dualstream":(.682,.731,.468,.796),
          "dca_sleep":(.665,.706,.407,.770), "v8":(.713,.745,.507,.775)},
 "cfs":  {"deepsleepnet":(.567,.674,.600,.613), "sleeppgnet":(.601,.700,.603,.666),
          "insightsleepnet":(.581,.692,.616,.662), "wang_dualstream":(.622,.719,.623,.699),
          "dca_sleep":(.547,.657,.537,.619), "v8":(.621,.718,.652,.676)}}
for coh, models in T3.items():
    print(f"\n=== TABLE III / {coh.upper()} ===")
    for m, (k,mf,df,rf) in models.items():
        r = folds(coh, m)
        if r is None: print(f"  MISSING  {m}"); bad += 1; continue
        chk(f"{m} kappa",   r["kappa"][0], k)
        chk(f"{m} macroF1", r["macro"][0], mf)
        chk(f"{m} DeepF1",  r["deep"][0],  df)
        chk(f"{m} REM F1",  r["rem"][0],   rf)

# ---- Table IV ---------------------------------------------------------
print("\n=== TABLE IV / CFS (mean and SD) ===")
for m, (k, sd) in {"v8":(.621,.023), "enc_dual_same":(.612,.015),
                   "enc_single_card":(.604,.026), "plain_transformer":(.619,.025),
                   "v8_full_with_ultradian":(.604,.024)}.items():
    r = folds("cfs", m)
    if r is None: print(f"  MISSING  {m}"); bad += 1; continue
    chk(f"{m} kappa", r["kappa"][0], k)
    chk(f"{m} SD",    r["kappa"][1], sd, tol=0.0006)

# ---- Section IV-D linear probe ---------------------------------------
print("\n=== SECTION IV-D / linear probe R2 ===")
P = {"mesa":{"HR_mean":.269,"HR_std":.506,"RMSSD":.447,"HF_power":.292,"resp_rate":-.030},
     "cfs" :{"HR_mean":.128,"HR_std":.159,"RMSSD":.143,"HF_power":.052,"resp_rate":-.029}}
df = pd.read_csv(R/"_paper_results/linear_probe_per_fold.csv")
for coh, targ in P.items():
    d = df[df.cohort == coh]
    print(f"  -- {coh.upper()} ({len(d)} folds; must be 5) --")
    if len(d) != 5: print("  FAIL  wrong fold count"); bad += 1
    for t, e in targ.items(): chk(f"{coh} {t}", d[t+"_r2"].mean(), e)

print(f"\n{'='*60}\n{ok} matched, {bad} mismatched\n{'='*60}")
