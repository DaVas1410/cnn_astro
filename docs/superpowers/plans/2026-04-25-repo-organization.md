# Repo Organization (Option A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply 5 surgical file-system fixes to reduce daily navigation friction in the cnn_astro research repo.

**Architecture:** Pure file moves, deletes, and renames — no code changes. No imports are broken because fractal_classifier_02 is a standalone package and the notebooks are self-contained.

**Tech Stack:** Bash (mv, rm, cp), Python not required.

---

### Task 1: Unnest fractal_classifier_02

**Files:**
- Move: `src/fractal_classifier_01/fractal_classifier_02/` → `src/fractal_classifier_02/`

- [ ] **Step 1: Copy the package to its new location**

```bash
cp -r /home/davas/Documents/cnn_astro/src/fractal_classifier_01/fractal_classifier_02 \
      /home/davas/Documents/cnn_astro/src/fractal_classifier_02
```

- [ ] **Step 2: Verify the copy is complete**

```bash
diff -rq \
  /home/davas/Documents/cnn_astro/src/fractal_classifier_01/fractal_classifier_02 \
  /home/davas/Documents/cnn_astro/src/fractal_classifier_02
```
Expected: no output (directories are identical)

- [ ] **Step 3: Remove the old nested copy**

```bash
rm -rf /home/davas/Documents/cnn_astro/src/fractal_classifier_01/fractal_classifier_02
```

- [ ] **Step 4: Verify new location exists and old is gone**

```bash
ls /home/davas/Documents/cnn_astro/src/fractal_classifier_02/
ls /home/davas/Documents/cnn_astro/src/fractal_classifier_01/fractal_classifier_02 2>&1
```
Expected: first lists package files; second prints "No such file or directory"

---

### Task 2: Remove empty placeholder experiment dirs

**Files:**
- Delete: `experiments/kmin/`, `experiments/kmax/`, `experiments/dual_output/`

- [ ] **Step 1: Confirm dirs are empty**

```bash
find /home/davas/Documents/cnn_astro/experiments/kmin \
     /home/davas/Documents/cnn_astro/experiments/kmax \
     /home/davas/Documents/cnn_astro/experiments/dual_output \
     -mindepth 1 | head -20
```
Expected: no output (all empty)

- [ ] **Step 2: Remove the empty dirs**

```bash
rmdir /home/davas/Documents/cnn_astro/experiments/kmin \
      /home/davas/Documents/cnn_astro/experiments/kmax \
      /home/davas/Documents/cnn_astro/experiments/dual_output
```

- [ ] **Step 3: Verify**

```bash
ls /home/davas/Documents/cnn_astro/experiments/
```
Expected: only `archive/` remains

---

### Task 3: Organize loose top-level notebooks

**Files:**
- Create: `notebooks/multi_param/`
- Move: 4 notebooks from `notebooks/` root into `notebooks/multi_param/`

- [ ] **Step 1: Create the subfolder**

```bash
mkdir /home/davas/Documents/cnn_astro/notebooks/multi_param
```

- [ ] **Step 2: Move the four notebooks**

```bash
mv /home/davas/Documents/cnn_astro/notebooks/multi_parameter_regression.ipynb \
   /home/davas/Documents/cnn_astro/notebooks/multi_param/

mv /home/davas/Documents/cnn_astro/notebooks/sigma_regression.ipynb \
   /home/davas/Documents/cnn_astro/notebooks/multi_param/

mv /home/davas/Documents/cnn_astro/notebooks/sigma_research.ipynb \
   /home/davas/Documents/cnn_astro/notebooks/multi_param/

mv /home/davas/Documents/cnn_astro/notebooks/flexible_dataset_generation.ipynb \
   /home/davas/Documents/cnn_astro/notebooks/multi_param/
```

- [ ] **Step 3: Verify notebooks root is clean**

```bash
ls /home/davas/Documents/cnn_astro/notebooks/
```
Expected: `comparison/  data_generation/  kmax_experiments/  kmin_experiments/  multi_param/`

```bash
ls /home/davas/Documents/cnn_astro/notebooks/multi_param/
```
Expected: all 4 notebooks listed

---

### Task 4: Fix requirements filename typo

**Files:**
- Rename: `requirements_py39.txt.txt` → `requirements_py39.txt`

- [ ] **Step 1: Rename the file**

```bash
mv /home/davas/Documents/cnn_astro/requirements_py39.txt.txt \
   /home/davas/Documents/cnn_astro/requirements_py39.txt
```

- [ ] **Step 2: Verify**

```bash
ls /home/davas/Documents/cnn_astro/requirements_py3*
```
Expected: `requirements_py311.txt  requirements_py39.txt`

---

### Task 5: Update README repository map

**Files:**
- Modify: `README.md` — Compact Repository Map section

- [ ] **Step 1: Replace the repository map block**

Find the `## Compact Repository Map` section and replace the code block with:

```text
cnn_astro/
├── src/
│   ├── dataset_generator.py
│   ├── param_sampler.py
│   ├── plot_performance.py
│   ├── fractal_classifier_01/
│   ├── fractal_classifier_02/
│   └── pyFC_lib/
├── scripts/
├── notebooks/
│   ├── data_generation/
│   ├── kmin_experiments/
│   ├── kmax_experiments/
│   ├── comparison/
│   └── multi_param/
├── data/
├── outputs/
├── configs/
├── tests/
├── docs/
└── experiments/
    └── archive/
```

- [ ] **Step 2: Also update the Project Status note about placeholder dirs**

Remove or update this line in the Project Status section:
> `Some folders are placeholders for future consolidation (docs/, experiments/kmin, experiments/kmax, experiments/dual_output).`

Replace with:
> `experiments/archive/ contains first-steps artifacts. docs/ is reserved for specs and plans.`

- [ ] **Step 3: Verify README renders correctly**

```bash
head -30 /home/davas/Documents/cnn_astro/README.md
grep -n "Compact Repository Map" /home/davas/Documents/cnn_astro/README.md
```
Expected: section header found at expected line number

---
