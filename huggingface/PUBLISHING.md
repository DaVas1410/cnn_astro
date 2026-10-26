# Publishing turbulens to Hugging Face

Everything is staged in this folder and checked locally. One script does the whole upload.

## What gets published

| Repo | Id | Contents | Size |
|---|---|---|---|
| Model | `NAME/turbulens-ensembles` | 5 ensembles (multitask + 4 single-target), config + checkpoint per member, laid out so a download works directly as `--output-root` | 2.2 GB |
| Dataset | `NAME/turbulens-synthetic-data` | train/val/test `.h5` files, dataset card, and the exact generator under `generation/` | 10.3 GB |
| Space | `NAME/turbulens` | Static landing page (free) linking the model, dataset, code and guide. The Gradio demo in `space/` needs HF PRO: publish it with `--space-kind gradio` | tiny |

## Steps

1. Create a **write** token at huggingface.co/settings/tokens and log in (you do this, the token
   never goes through anyone else):

   ```bash
   hf auth login
   ```

2. Check everything without uploading:

   ```bash
   python huggingface/publish.py --user NAME --dry-run
   ```

3. Publish:

   ```bash
   python huggingface/publish.py --user NAME
   ```

   Options: `--only model dataset space` to pick parts, `--private` to create private repos.
   If an upload is interrupted, rerun the same command; unchanged files are skipped.

4. Open the three URLs it prints. The Space takes a few minutes to build on first start
   (it installs `turbulens` from GitHub and downloads the ensembles).

5. Link the Space and model repo from `README.md` and `GUIDE.md`.

## Notes

- Published so far (private): model, dataset and static Space under `Davas1410`. Make each public in its Settings when ready.
- Placeholders (`<MODEL_REPO_ID>`, etc.) in the cards and `space/app.py` are filled in at upload
  time. The files in this folder are not modified.
- The Space installs `turbulens` from the public GitHub repo, so GitHub `main` must contain the
  version you want the Space to run.
- To test the Space locally before publishing: import `huggingface/space/app.py`, set
  `app._ENSEMBLE_ROOT = Path("outputs")`, and call `app.predict(...)`.
