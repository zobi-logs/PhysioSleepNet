# PhysioSleepNet

Interpretable PPG-only sleep staging via autonomic representation learning.
Code for the paper *"PhysioSleepNet: Interpretable Sleep Staging from
Continuous PPG via Autonomic Representation Learning."*

## What is and is not included

Included: model definitions, the training/evaluation harness, the frozen
cross-validation splits, per-fold metrics, and the scripts that produce
every figure and table in the paper.

**Not included, and not redistributable:** the MESA and CFS
polysomnography recordings and the NSRR harmonized clinical covariates
(AHI, age, BMI, diabetes). Both are licensed per user by the National
Sleep Research Resource (https://sleepdata.org). Request access there;
`manifest_mesa_ppg.csv` lists the exact recording IDs we used.

Model checkpoints are also excluded for size.

## Layout

    models.py                    model registry; PhysioSleepNet + 5 baselines
    encoder_ablation.py          parameter-matched encoder variants
    harness.py                   training and 5-fold evaluation
    ablation_result_analysis.py  reproduces Table IV
    phenotype_analysis.py        AHI / diabetes stratification (Fig. 6)
    linear_probe.py              latent -> HRV ridge probes (Fig. 3)
    figure_0*.py                 all paper figures
    folds/                       frozen subject-level splits, seed 42
    benchmark_results/           per-fold metrics.json and paper outputs

## Reproducing

All reported results use a single training seed (42) and five
subject-disjoint folds. Splits are written once to `folds/*.json` and
reused by every model, so all comparisons are on identical partitions.

    # train one model on one cohort
    python harness.py --model v8 --cohort cfs --seeds 42 --folds 0 1 2 3 4

    # reproduce the ablation table
    python ablation_result_analysis.py

    # regenerate figures
    python figure_01_linear_probe.py
    python figure_06_confusion.py

Runs are resume-safe: a fold directory containing `DONE` is skipped.

## Notes

- `per_subject_{mesa,cfs}_metrics.csv` give per-subject Cohen's kappa.
  The CFS file has 242 rows, not 305 - the phenotype analysis is
  restricted to subjects with a harmonized AHI available. Rejoin
  covariates from your own NSRR download to reproduce Fig. 6.
- Wang-DualStream is run from the authors' repository
  (https://github.com/DavyWJW/sleep-staging-models), which is not
  vendored here. Clone it alongside this repo.
- Trained on NVIDIA RTX A6000 (48 GB), ~4 h/fold on MESA, ~1.5 h/fold on CFS.

## Citation

    [BibTeX once published]

## License

MIT (see LICENSE). This covers the code in this repository. The baseline
architectures are reimplementations of published work by their respective
authors, cited in the paper; Wang-DualStream is run from the authors' own
repository and is not redistributed here. The MESA and CFS data are governed
by the NSRR data use agreement.
