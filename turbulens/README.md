# turbulens

Predicts fractal-turbulence parameters (`k_min`, `k_max`, `sigma`, `beta`) from a 2D radio-astronomy
image or FITS cube, using a pretrained deep-ensemble CNN, or classical (non-CNN) spectral estimators
that need no trained model at all. Full docs (physical background, CLI reference, API): `docs/turbulens/`
(build with `sphinx-build docs/turbulens docs/turbulens/_build`, or read the `.rst` sources directly).

## Install

```bash
pip install -e ".[image]"   # from a checkout of this repository
```

Requires Python >=3.10. `turbulens` itself ships no trained model — see "Getting a model" below.

## Quickstart: classical estimate, no model needed

Works immediately after install, on any FITS cube or plain image, real or synthetic:

```bash
# Summarize shape, WCS, spectral coverage, pixel stats
turbulens inspect --input observation.fits

# Classical power-law fit (appropriate for real observational data with a continuous spectrum)
turbulens infer --input observation.fits --integrate 0:100 --method fft-powerlaw

# Classical band-edge detection (appropriate for synthetic, sharply band-limited pyFC images only)
turbulens infer --input synthetic_slice.npy --method fft-band
```

## Quickstart: CNN prediction, needs a trained ensemble

```bash
turbulens infer --input observation.fits --integrate 0:100 --real-data \
    --method cnn --model auto --output-root <output_root> --out predictions.ecsv
```

- **`--real-data` is required for any real telescope input.** It recenters the image onto the
  training-set's own normalization scale instead of the image's own pixel range. Omitting it on real
  data silently produces confident-looking but meaningless numbers. Leave it off for synthetic pyFC
  images, which are already on the training log-density scale.
- **`--model auto --output-root <output_root>`** resolves the newest ensemble under `<output_root>`
  automatically; add `--target {k_min,k_max,sigma,beta}` to use that target's dedicated single-target
  ensemble instead of the multitask one. `--model /path/to/ensemble/members` points at one directly.

### Getting a model

`turbulens` is inference code only — it does not ship a trained ensemble. Two ways to get one:

1. **Download a pretrained ensemble** from the project's Hugging Face model repo (see the main
   [repository README](../README.md) for the link), then point `--output-root` at wherever you
   downloaded it.
2. **Train your own** with `turbulens/training/` (see its own README) — a YAML-driven pipeline that
   produces the same `<output_root>/<multitask|single_<target>>/<version>/ensemble/members/` layout
   `--model auto` expects.

## Using it as a library

```python
from turbulens.io.image_loaders import load_input
from turbulens.inference import Inferencer

cube = load_input("observation.fits")
image = cube.project(0, 100)  # or cube.slice_channel(i), or cube.project_velocity(lo, hi)
table = Inferencer("/path/to/ensemble/members").predict(image.data, real_data=True)
```

See `docs/turbulens/api/index.rst` for the full API reference (`Cube`, `Inferencer`,
`EnsembleRegistry`, `SpectralEstimator`, and more).

## Why some numbers need care

- **`--real-data` on real input, always.** See above.
- **`fft-band` is for synthetic data, `fft-powerlaw` is for real data** — they measure different
  spectral features and are not interchangeable; see `docs/turbulens/concepts.rst`.
- **A prediction near a target's training range edge may be clamped, not measured** — the model's raw
  output is clipped to `[0, 1]` before conversion to physical units, so a genuinely out-of-range input
  can silently report the boundary value instead of extrapolating.
- **`Inferencer.predict` warns** when a large fraction of input pixels hit the normalization clip
  boundary — a sign the input's intensity range doesn't match what the model was trained on.
