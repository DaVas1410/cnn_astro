# Publishing turbulens to Hugging Face

Staged locally in this directory: `model_card.md`, `dataset_card.md`, `space/` (Gradio app +
GASS examples). Nothing here has been pushed yet. Steps to actually publish, once you're ready:

## 0. Fill in the placeholders

Replace `<your-hf-username>` / `<MODEL_REPO_ID>` / `<DATASET_REPO_ID>` in `model_card.md`,
`dataset_card.md`, and `space/app.py` with your real repo IDs (e.g. `yourname/turbulens-ensembles`,
`yourname/turbulens-synthetic-data`).

## 1. Authenticate

```bash
huggingface-cli login    # or: set HF_TOKEN env var
```

## 2. Create and push the model repo

Only the files `load_member` actually reads are needed per member: `configuration.json`,
`COMPLETED.json`, `checkpoints/best_checkpoint.pt` (not the plots/predictions/logs/history —
those stay local). ~2.1 GB total across all 5 ensembles (25 members).

```python
from huggingface_hub import HfApi, create_repo
import shutil, pathlib

REPO_ID = "yourname/turbulens-ensembles"
create_repo(REPO_ID, repo_type="model", exist_ok=True)

staging = pathlib.Path("hf_staging_model")
for mode_dir in ["multitask", "single_k_min", "single_k_max", "single_sigma", "single_beta"]:
    src_root = pathlib.Path("outputs") / mode_dir / "local_v2" / "ensemble" / "members"
    for member in src_root.iterdir():
        dst = staging / mode_dir / "local_v2" / "ensemble" / "members" / member.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(member / "configuration.json", dst / "configuration.json")
        shutil.copy(member / "COMPLETED.json", dst / "COMPLETED.json")
        (dst / "checkpoints").mkdir(exist_ok=True)
        shutil.copy(member / "checkpoints" / "best_checkpoint.pt", dst / "checkpoints" / "best_checkpoint.pt")
shutil.copy("huggingface/model_card.md", staging / "README.md")

HfApi().upload_folder(folder_path=str(staging), repo_id=REPO_ID, repo_type="model")
```

## 3. Create and push the dataset repo

```python
from huggingface_hub import HfApi, create_repo
import shutil

REPO_ID = "yourname/turbulens-synthetic-data"
create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

api = HfApi()
api.upload_file(path_or_fileobj="huggingface/dataset_card.md", path_in_repo="README.md",
                 repo_id=REPO_ID, repo_type="dataset")
for split, fname in [
    ("train", "flat_4param_128x128_100000_disjoint_train.h5"),
    ("val", "flat_4param_128x128_20000_disjoint_val.h5"),
    ("test", "flat_4param_128x128_50000_disjoint_test.h5"),
]:
    api.upload_file(
        path_or_fileobj=f"data/raw/{fname}", path_in_repo=f"{split}/{fname}",
        repo_id=REPO_ID, repo_type="dataset",
    )
```

## 4. Create and push the Space

```bash
huggingface-cli repo create yourname/turbulens --type space --space_sdk gradio
```

```python
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="huggingface/space", repo_id="yourname/turbulens", repo_type="space")
```

The Space's `app.py` downloads the model repo at startup via `snapshot_download` — make sure
`MODEL_REPO_ID` in `app.py` is filled in and step 2 is pushed before the Space is opened.

## 5. Wire the links back into the repo

Once pushed, update:
- Root `README.md`'s "Just want to run inference?" section with a link to the Space/model repo.
- `turbulens/README.md`'s "Getting a model" section with the real model repo link.
- `docs/turbulens/index.rst`'s "Getting an ensemble" section to mention the HF download option
  alongside training your own.
