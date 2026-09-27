# turbulens

Predicts fractal-turbulence parameters (`k_min`, `k_max`, `sigma`, `beta`) from a radio-astronomy
FITS cube or a synthetic 2D image, using a pretrained deep-ensemble CNN, or classical spectral
estimators that need no trained model at all.

## Install

Requires Python >= 3.10.

```bash
pip install -e ".[image]"
```

## Try it in three commands

No trained model needed for these — they use a bundled real HI radio observation and a classical
spectral estimator.

**1. Inspect the file** (shape, WCS, velocity range, pixel stats):

```bash
turbulens inspect --input huggingface/space/examples/gass_0_0_1774393774.fits.gz
```

```
Shape: (1201, 125, 125) (n_channels=1201)
Unit: K
Velocity range: -494.70 to 494.70 km/s (CTYPE=VOPT)
Pixel stats: min=-11.9, max=148.4, mean=3.1, std=13.0, nan_fraction=0.0000
```

**2. Integrate the full velocity range into one 2D image and predict `k_min`/`k_max`** with the
classical power-law estimator (appropriate for real data, which has a continuous spectrum rather
than a sharp cutoff):

```bash
turbulens infer --input huggingface/space/examples/gass_0_0_1774393774.fits.gz \
    --integrate 0:1201 --method fft-powerlaw
```

```
image_index   target        value        epistemic_std
----------- --------- ------------------ -------------
          0     k_min                2.0           nan
          0     k_max               61.0           nan
          0     alpha -4.023437648868689           nan
          0 r_squared 0.9959089502469131           nan
```

**3. Try your own file** the same way — FITS cube, or a plain PNG/TIFF/`.npy` image:

```bash
turbulens inspect --input your_file.fits
turbulens infer --input your_file.fits --integrate LOW:HIGH --method fft-powerlaw
```

### CNN inference (needs a pretrained ensemble)

The classical estimator above needs no model; the CNN ensemble is more accurate but needs one.

**If you have this repo's own `outputs/` locally** (the trained ensembles are git-ignored, not
part of the git checkout, but present if you're on a machine where they were produced or synced),
point `--output-root` straight at it:

```bash
turbulens infer --input your_file.fits --integrate LOW:HIGH --real-data \
    --method cnn --model auto --output-root outputs
```

```
image_index target        value           epistemic_std
----------- ------ ------------------- -------------------
          0  k_min  1.3792097568511963  0.5257347226142883
          0  k_max   63.90980911254883 0.15796305239200592
          0  sigma    4.14192008972168  0.1170581504702568
          0   beta -2.5287206172943115 0.12315365672111511
```

**Otherwise**, download the pretrained ensemble from the
[`tutor-ensemble-v1` release](https://github.com/wbandabarragan/cnn-turbulence/releases/tag/tutor-ensemble-v1)
(or train your own, see below), unzip it, and point `--output-root` at wherever it landed:

```bash
unzip turbulens-tutor-ensemble.zip -d turbulens-ensemble
turbulens infer --input your_file.fits --integrate LOW:HIGH --real-data \
    --method cnn --model auto --output-root turbulens-ensemble
```

Either way, `--model auto` finds the ensemble under `--output-root` automatically (newest version
of the multitask ensemble); add `--target {k_min,k_max,sigma,beta}` to use that target's dedicated
single-target ensemble instead, if one is available under the same root.

`--real-data` is required for real telescope data (recenters onto the training scale) — see
`turbulens infer --help` for why, and [`turbulens/README.md`](turbulens/README.md) for the full
quickstart. Full docs (physical background, CLI reference, API) are in
[`docs/turbulens/`](docs/turbulens/index.rst).

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
