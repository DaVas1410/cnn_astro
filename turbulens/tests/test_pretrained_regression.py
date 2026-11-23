"""
Regression test of the real pretrained multitask ensemble on a real GASS cube.

The rest of the suite exercises the CNN path with tiny, randomly initialized
ensembles; this test checks that the *published* weights, loaded and
preprocessed through the full ``turbulens infer`` path, still give the
reference predictions. It needs the ~450 MB ensemble, so it only runs when
``TURBULENS_PRETRAINED`` is set:

- ``TURBULENS_PRETRAINED=hub`` uses ``--model auto`` (the Hugging Face
  ensemble; served from the local cache if present, downloaded otherwise,
  which needs access to the private repo);
- ``TURBULENS_PRETRAINED=<output_root>`` uses a training-pipeline output
  root, as ``--model auto --output-root <output_root>``.
"""

import os
from pathlib import Path

import pytest
from astropy.table import QTable

from turbulens.cli import main

PRETRAINED = os.environ.get("TURBULENS_PRETRAINED")
FIXTURE = Path(__file__).resolve().parents[1] / "io" / "tests" / "data" / "gass_0_0_1774393774.fits.gz"

# Multitask ensemble, Hugging Face revision a784ee29 (identical to the local_v2 training run),
# on the full velocity range of the GASS (l, b) = (0, 0) field: (value, epistemic_std).
REFERENCE = {
    "k_min": (1.3792, 0.5257),
    "k_max": (63.9098, 0.1580),
    "sigma": (4.1419, 0.1171),
    "beta": (-2.5287, 0.1232),
}

pytestmark = pytest.mark.skipif(
    not PRETRAINED, reason="set TURBULENS_PRETRAINED=hub or =<output_root> to run against the real ensemble"
)


def test_pretrained_multitask_ensemble_reproduces_reference_on_gass(tmp_path):
    out_path = tmp_path / "predictions.ecsv"
    model_args = ["--model", "auto"]
    if PRETRAINED != "hub":
        model_args += ["--output-root", PRETRAINED]

    main([
        "infer", "--input", str(FIXTURE), "--integrate", "0:1201", "--real-data",
        *model_args, "--out", str(out_path),
    ])

    table = QTable.read(out_path)
    predictions = {str(row["target"]): (float(row["value"]), float(row["epistemic_std"])) for row in table}
    assert set(predictions) == set(REFERENCE)
    for target, (value, std) in REFERENCE.items():
        assert predictions[target][0] == pytest.approx(value, abs=0.01), target
        assert predictions[target][1] == pytest.approx(std, abs=0.01), target
    assert table.meta["n_members"] == 5
