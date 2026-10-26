"""Publish turbulens to the Hugging Face Hub: model repo, dataset repo, and Space.

    hf auth login                                   # once, with a write token
    python huggingface/publish.py --user NAME --dry-run
    python huggingface/publish.py --user NAME       # publish everything
    python huggingface/publish.py --user NAME --only model space

Repo ids created: NAME/turbulens-ensembles (model), NAME/turbulens-synthetic-data (dataset),
NAME/turbulens (Space). Placeholders in the cards and app.py are substituted at upload time; the
files in this folder are never modified. Re-running is safe: unchanged files are skipped.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF_DIR = ROOT / "huggingface"
VERSION = "local_v2"
MODES = ["multitask", "single_k_min", "single_k_max", "single_sigma", "single_beta"]
MEMBER_FILES = ["configuration.json", "COMPLETED.json", "checkpoints/best_checkpoint.pt"]
DATASET_FILES = {
    "train": "flat_4param_128x128_100000_disjoint_train.h5",
    "val": "flat_4param_128x128_20000_disjoint_val.h5",
    "test": "flat_4param_128x128_50000_disjoint_test.h5",
}
GENERATION_FILES = [
    "scripts/run_generation_balanced.py",
    "scripts/gen_splits_disjoint.sh",
    "src/dataset_generator.py",
]


def substitute(text: str, ids: dict[str, str]) -> str:
    text = text.replace("<MODEL_REPO_ID>", ids["model"]).replace("<REPO_ID>", ids["model"])
    text = text.replace("<DATASET_REPO_ID>", ids["dataset"])
    return text.replace("<your-hf-username>/turbulens-ensembles", ids["model"])


def model_patterns() -> list[str]:
    return [f"{m}/{VERSION}/ensemble/members/*/{f}" for m in MODES for f in MEMBER_FILES]


def model_files() -> list[Path]:
    out = ROOT / "outputs"
    pats = model_patterns()
    return sorted(
        p for p in out.rglob("*")
        if p.is_file() and any(fnmatch.fnmatch(p.relative_to(out).as_posix(), g) for g in pats)
    )


def generation_files() -> list[tuple[Path, str]]:
    arch = ROOT / "archive"
    files = [(arch / f, f) for f in GENERATION_FILES]
    pyfc = arch / "src" / "pyFC_lib" / "pyFC"
    for p in sorted(pyfc.glob("*.py")):
        if p.name != "test_clouds.py":
            files.append((p, f"src/pyFC_lib/pyFC/{p.name}"))
    return files


def preflight() -> list[str]:
    problems = []
    members = model_files()
    expected = len(MODES) * 5 * len(MEMBER_FILES)
    if len(members) != expected:
        problems.append(f"model: expected {expected} member files under outputs/, found {len(members)}")
    for split, name in DATASET_FILES.items():
        if not (ROOT / "data" / "raw" / name).is_file():
            problems.append(f"dataset: missing data/raw/{name}")
    for src, _ in generation_files():
        if not src.is_file():
            problems.append(f"dataset: missing generator file {src.relative_to(ROOT)}")
    for needed in ["model_card.md", "dataset_card.md", "space/app.py", "space/README.md",
                   "static_space/index.html", "static_space/README.md"]:
        if not (HF_DIR / needed).is_file():
            problems.append(f"missing huggingface/{needed}")
    return problems


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def dry_run(ids: dict[str, str], only: set[str]) -> None:
    if "model" in only:
        files = model_files()
        print(f"[model]   {ids['model']}: {len(files)} files, {gb(sum(f.stat().st_size for f in files))}")
    if "dataset" in only:
        sizes = [(ROOT / "data" / "raw" / n).stat().st_size for n in DATASET_FILES.values()]
        print(f"[dataset] {ids['dataset']}: 3 .h5 files ({gb(sum(sizes))}) + card + generation/ scripts")
    if "space" in only:
        total = sum(p.stat().st_size for p in (HF_DIR / "space").rglob("*") if p.is_file())
        print(f"[space]   {ids['space']}: {gb(total)} (app, README, requirements, 4 GASS examples)")


def publish_model(api, ids, private):
    from huggingface_hub import create_repo

    create_repo(ids["model"], repo_type="model", private=private, exist_ok=True)
    card = substitute((HF_DIR / "model_card.md").read_text(encoding="utf-8"), ids)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=ids["model"],
                    repo_type="model", commit_message="Add model card")
    api.upload_folder(folder_path=str(ROOT / "outputs"), repo_id=ids["model"], repo_type="model",
                      allow_patterns=model_patterns(), commit_message="Add ensembles")


def publish_dataset(api, ids, private):
    from huggingface_hub import create_repo

    create_repo(ids["dataset"], repo_type="dataset", private=private, exist_ok=True)
    card = substitute((HF_DIR / "dataset_card.md").read_text(encoding="utf-8"), ids)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=ids["dataset"],
                    repo_type="dataset", commit_message="Add dataset card")
    with tempfile.TemporaryDirectory() as tmp:
        for src, rel in generation_files():
            dst = Path(tmp) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        api.upload_folder(folder_path=tmp, path_in_repo="generation", repo_id=ids["dataset"],
                          repo_type="dataset", commit_message="Add data generator")
    for split, name in DATASET_FILES.items():
        print(f"  uploading {split}/{name} ...", flush=True)
        api.upload_file(path_or_fileobj=str(ROOT / "data" / "raw" / name), path_in_repo=f"{split}/{name}",
                        repo_id=ids["dataset"], repo_type="dataset", commit_message=f"Add {split} split")


def publish_space(api, ids, private, kind):
    from huggingface_hub import create_repo

    src, entry = (HF_DIR / "static_space", "index.html") if kind == "static" else (HF_DIR / "space", "app.py")
    create_repo(ids["space"], repo_type="space", space_sdk=kind if kind == "static" else "gradio",
                private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(src, tmp, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        page = Path(tmp) / entry
        page.write_text(substitute(page.read_text(encoding="utf-8"), ids), encoding="utf-8")
        api.upload_folder(folder_path=tmp, repo_id=ids["space"], repo_type="space",
                          commit_message=f"Add {kind} Space")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="HF username or org that will own the repos")
    ap.add_argument("--only", nargs="+", choices=["model", "dataset", "space"],
                    default=["model", "dataset", "space"])
    ap.add_argument("--private", action="store_true", help="create repos as private (default: public)")
    ap.add_argument("--space-kind", choices=["static", "gradio"], default="static",
                    help="static landing page (free, default) or the Gradio demo (needs HF PRO)")
    ap.add_argument("--dry-run", action="store_true", help="check files and print what would be uploaded")
    args = ap.parse_args()

    ids = {
        "model": f"{args.user}/turbulens-ensembles",
        "dataset": f"{args.user}/turbulens-synthetic-data",
        "space": f"{args.user}/turbulens",
    }
    only = set(args.only)

    problems = preflight()
    if problems:
        print("Preflight failed:\n  " + "\n  ".join(problems))
        return 1
    if args.dry_run:
        dry_run(ids, only)
        print("Preflight OK (dry run, nothing uploaded).")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        me = api.whoami()["name"]
    except Exception:
        print("Not logged in. Run `hf auth login` with a write token first.")
        return 1
    print(f"Logged in as {me}; publishing to {args.user} ({'private' if args.private else 'public'}).")

    if "model" in only:
        print("Model repo ..."); publish_model(api, ids, args.private)
    if "dataset" in only:
        print("Dataset repo ..."); publish_dataset(api, ids, args.private)
    if "space" in only:
        print("Space ..."); publish_space(api, ids, args.private, args.space_kind)
    for kind, rid in ids.items():
        if kind in only:
            prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}[kind]
            print(f"https://huggingface.co/{prefix}{rid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
