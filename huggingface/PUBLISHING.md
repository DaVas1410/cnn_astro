# Publishing turbulens to Hugging Face

Everything is staged in this folder and checked locally. One script does the whole upload.

## What gets published

| Repo | Id | Contents | Size |
|---|---|---|---|
| Model | `NAME/turbulens-ensembles` | 5 ensembles (multitask + 4 single-target), config + checkpoint per member, laid out so a download works directly as `--output-root` | 2.2 GB |
| Dataset | `NAME/turbulens-synthetic-data` | train/val/test `.h5` files, dataset card, and the exact generator under `generation/` | 10.3 GB |

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

   Options: `--only model dataset` to pick parts, `--private` to create private repos.
   If an upload is interrupted, rerun the same command; unchanged files are skipped.

4. Open the two URLs it prints.

5. Link the model and dataset repos from `README.md`.

## Notes

- Published (private): model and dataset under `cPhysPlus`. Make each public in its Settings when ready.
- Placeholders (`<MODEL_REPO_ID>`, etc.) in the cards are filled in at upload time. The files in this folder are not modified.
- There is no Space: only the model and dataset are published.
