# Trying turbulens

A quick, hands-on way to run what I built yourself. Should take a few minutes.

## What this is

`turbulens` predicts four physical parameters of a turbulent gas cloud (`k_min`, `k_max`, `sigma`,
`beta`) from a 2D image or a radio telescope observation, using a deep-ensemble CNN trained on
synthetic data. It also ships two classical, non-ML estimators for comparison.

## What you'll need

- Python 3.10+
- This repo, cloned or unzipped locally
- The pretrained ensemble, if you want to try the CNN model (the classical methods below don't
  need it): download `turbulens-tutor-ensemble.zip` from the
  [`tutor-ensemble-v1` release](https://github.com/wbandabarragan/cnn-turbulence/releases/tag/tutor-ensemble-v1)

## Install

From the repo root:

```bash
pip install -e ".[image]"
```

## Look at a real observation

The repo ships a real HI radio observation (Parkes GASS survey) you can try right away:

```bash
turbulens inspect --input huggingface/space/examples/gass_0_0_1774393774.fits.gz
```

```
Shape: (1201, 125, 125) (n_channels=1201)
Unit: K
Velocity range: -494.70 to 494.70 km/s (CTYPE=VOPT)
Pixel stats: min=-11.9, max=148.4, mean=3.1, std=13.0, nan_fraction=0.0000
```

That's a 125x125 image with 1201 velocity channels. `turbulens` needs a single 2D image, so the
next steps integrate (sum) the channels into one.

## Classical estimate, no model needed

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

This fits a power law to the image's Fourier spectrum and reads `k_min`/`k_max` off as the scales
where the spectrum departs from that line. No trained model involved, works on any real
observation.

## Run the CNN ensemble

Unzip `turbulens-tutor-ensemble.zip` somewhere and point `--output-root` at it:

```bash
turbulens infer --input huggingface/space/examples/gass_0_0_1774393774.fits.gz \
    --integrate 0:1201 --real-data --method cnn --model auto \
    --output-root /path/to/unzipped/turbulens-tutor-ensemble
```

```
image_index target        value           epistemic_std
----------- ------ ------------------- -------------------
          0  k_min  1.3792097568511963  0.5257347226142883
          0  k_max   63.90980911254883 0.15796305239200592
          0  sigma    4.14192008972168  0.1170581504702568
          0   beta -2.5287206172943115 0.12315365672111511
```

`--real-data` matters here: it tells `turbulens` this is a real telescope image, not a synthetic
training-style image, and recenters it onto the scale the model expects. `epistemic_std` is the
ensemble's spread across its 5 independently trained members, not a calibrated confidence interval
(see the model card's limitations section).

## Your own data

Any FITS cube, or a plain PNG/TIFF/`.npy` image, works the same way:

```bash
turbulens inspect --input your_file.fits
turbulens infer --input your_file.fits --integrate LOW:HIGH --real-data \
    --method cnn --model auto --output-root /path/to/unzipped/turbulens-tutor-ensemble
```

Run `inspect` first to see the channel count and velocity range, that tells you what `LOW:HIGH`
makes sense.

## If something breaks

`turbulens --help`, `turbulens infer --help`, and `turbulens inspect --help` cover every option.
Errors are meant to be readable, not raw tracebacks, so if you hit one that isn't, that's a bug,
let me know. Full docs (physics background, every CLI flag, the API) live in
[`docs/turbulens/`](docs/turbulens/index.rst), and [`README.md`](README.md) has the project
overview.
