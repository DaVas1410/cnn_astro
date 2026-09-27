# turbulens

Predicts fractal-turbulence parameters (`k_min`, `k_max`, `sigma`, `beta`) from a radio-astronomy
FITS cube or a synthetic 2D image, using a pretrained deep-ensemble CNN, or classical spectral
estimators that need no trained model at all.

## Quickstart

```bash
pip install -e ".[image]"
turbulens inspect --input observation.fits
turbulens infer --input observation.fits --integrate 0:100 --real-data --method fft-powerlaw
```

See [`turbulens/README.md`](turbulens/README.md) for the full quickstart (including CNN inference
with a pretrained ensemble) and [`docs/turbulens/`](docs/turbulens/index.rst) for the complete docs
(physical background, CLI reference, API).

## Repository layout

```text
turbulens/            # the package itself -- CLI (`turbulens infer`/`inspect`), inference, models,
                       # io, spectral estimators, and turbulens/training/ (the ensemble training pipeline)
docs/turbulens/        # turbulens' own Sphinx docs (source for the above)
huggingface/           # model card, dataset card, and Gradio Space for the HF releases
                       # (see huggingface/PUBLISHING.md)
notebooks/             # the two active notebooks -- see notebooks/NOTEBOOKS.md
data/, outputs/         # local datasets and trained ensemble checkpoints (git-ignored, large)
archive/               # everything superseded: the legacy pyFC dataset-generation pipeline
                       # (src/, scripts/, configs/, tests/), earlier notebook experiments,
                       # observational-image originals, and unpublished paper/thesis drafts.
                       # Not needed to use or develop turbulens -- kept for provenance only.
```

## Training your own ensemble

`turbulens/training/` is a YAML-driven, resumable pipeline that produces the
`<output_root>/<multitask|single_<target>>/<version>/ensemble/members/` layout `turbulens infer
--model auto` expects. See its own README for setup and usage.

## Provenance

This started as a broader research repo (synthetic dataset generation, multiple modeling
frameworks, per-target experiment notebooks) before converging on `turbulens` as the packaged,
production system. That history lives in `archive/` rather than in the working tree — see
`archive/docs/RESEARCH_JOURNEY.md` for the full narrative of how the project got here (model
iterations, a `k_max` labeling-bug fix, uncertainty methods).
