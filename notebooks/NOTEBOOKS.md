# Notebook Index

Only the three active `turbulens`-era notebooks live here now. Everything earlier (data-generation
notebooks, the per-target k_min/k_max/sigma experiment tracks, and the full model-comparison
history) has moved to `archive/`, grouped by project stage — see `archive/README.md`.

| Notebook | Parameter(s) | Approach | Status | Key Result |
|----------|-------------|----------|--------|------------|
| [turbulens_multitask_results.ipynb](turbulens_multitask_results.ipynb) | k_min, k_max, sigma, beta | `turbulens` multitask ensemble (production `local_v2` release) + multitask-vs-single-target comparison + real GASS HI inference | `complete` | Multitask test R2 = 0.9997 (k_min), 0.9795 (k_max), 0.9990 (sigma), 0.9203 (beta); single-target models edge out multitask on 3/4 targets at ~5x training cost, multitask wins on beta (`outputs/comparison/local_v2/`) |
| [turbulens_user_walkthrough.ipynb](turbulens_user_walkthrough.ipynb) | k_min, k_max, sigma, beta | User tutorial of the turbulens package: load/inspect a FITS cube, classical estimators, pretrained multitask and single-target CNN ensembles, performance on held-out synthetic data, characteristics on real data (spread, range limits, --real-data), method comparison on the four GASS fields, saving results | `active` | Runs in about 1 min on CPU; on GASS the CNN gives sigma 0.22-0.73, k_max at the resolution limit, beta near the -3 range edge |
| [paper_figures.ipynb](paper_figures.ipynb) | k_min, k_max, sigma, beta | Reproduces the manuscript's Figures 1/2/3/5 + residual-correlation and beta-vs-bandwidth analysis + live GASS re-inference | `active` | Figure 4 (worked examples) blocked: needs real test-image pixels, which live only on a remote host not present in this repo |
