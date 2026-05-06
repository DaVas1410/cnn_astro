# Deletion & Archive Log

This file records every file or directory removed or archived from this workspace, with the date and reason. It serves as the audit trail for the thesis/paper project.

---

## 2026-05-05

### Deleted

| Path | Reason |
|------|--------|
| `notebooks/kmin_experiments/fractal_classifier_v02.ipynb` | JSON corruption — file unreadable and unrecoverable |
| `notebooks/kmax_experiments/kmax_binned_classification.ipynb` | JSON corruption — file unreadable and unrecoverable |
| `experiments/kmin/` | Empty placeholder directory, never used |
| `experiments/kmax/` | Empty placeholder directory, never used |
| `experiments/dual_output/` | Empty placeholder directory, never used |

### Archived

| Original Path | Archive Path | Reason |
|--------------|-------------|--------|
| `scripts/run_generation.py` | `scripts/archive/run_generation.py` | Legacy — discrete k_min sweep; superseded by flexible workflow |
| `scripts/run_generation_random_both.py` | `scripts/archive/run_generation_random_both.py` | Legacy — random k_min+k_max; superseded by flexible workflow |
| `scripts/run_generation_random_kmin.py` | `scripts/archive/run_generation_random_kmin.py` | Legacy — random k_min only; superseded by flexible workflow |
| `scripts/run_generation_random_kmax.py` | `scripts/archive/run_generation_random_kmax.py` | Legacy — random k_max only; superseded by flexible workflow |
| `scripts/run_generation_flexible_kmax.py` | `scripts/archive/run_generation_flexible_kmax.py` | Specialized kmax variant; superseded by main flexible script |
| `scripts/job.sh` | `scripts/archive/job.sh` | Legacy SLURM array job; superseded by job_flexible.sh |
| `notebooks/kmin_experiments/pytorch_resnet.ipynb` | `notebooks/archive/pytorch_resnet.ipynb` | Only 33% executed — patch extraction approach abandoned |
| `notebooks/multi_parameter_regression.ipynb` | `notebooks/archive/multi_parameter_regression.ipynb` | 0% executed — multi-output (kmin+kmax+sigma) not yet pursued |
| `notebooks/compare_models_gradcam.ipynb` | `notebooks/archive/compare_models_gradcam.ipynb` | 73% executed — Grad-CAM comparison incomplete |
| `outputs/kmin/v02/` | `outputs/archive/kmin_v02/` | Early TF/Keras classification (6 k-values); superseded by regression approach |
| `outputs/kmin/v03/` | `outputs/archive/kmin_v03/` | Iteration on v02; superseded by regression approach |
| `outputs/kmin/v1/` | `outputs/archive/kmin_v1/` | Early classification baseline; superseded by regression approach |
| `outputs/kmin/resnet50_patches/` | `outputs/archive/kmin_resnet50_patches/` | Patch-based ResNet50 experiment; superseded by full-image regression |
