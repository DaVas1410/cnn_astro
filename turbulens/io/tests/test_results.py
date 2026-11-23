import warnings

import numpy as np
import pytest
from astropy.table import MaskedColumn, QTable

from turbulens.io.results import write_results


def _table():
    table = QTable()
    table["target"] = ["k_min"]
    table["value"] = [1.5]
    table["epistemic_std"] = [0.1]
    table.meta["clipped_pixel_fraction_max"] = 0.0
    return table


def test_write_results_ecsv_round_trips_data_and_metadata(tmp_path):
    path = tmp_path / "predictions.ecsv"
    write_results(_table(), path)

    read_back = QTable.read(path)
    assert list(read_back["target"]) == ["k_min"]
    assert read_back.meta["clipped_pixel_fraction_max"] == 0.0


def test_write_results_csv_warns_about_lost_metadata(tmp_path):
    path = tmp_path / "predictions.csv"

    with pytest.warns(UserWarning):
        write_results(_table(), path)

    read_back = QTable.read(path, format="ascii.csv")
    assert list(read_back["target"]) == ["k_min"]


def test_write_results_rejects_unsupported_suffix(tmp_path):
    with pytest.raises(ValueError):
        write_results(_table(), tmp_path / "predictions.txt")


def test_write_results_writes_uncomputed_values_as_empty_cells(tmp_path):
    table = _table()
    table["epistemic_std"] = [np.nan]
    path = tmp_path / "predictions.ecsv"

    write_results(table, path)

    data_row = next(line for line in path.read_text().splitlines() if line.startswith("k_min"))
    assert data_row == "k_min 1.5 \"\""
    read_back = QTable.read(path)
    assert read_back["epistemic_std"].mask[0]
    assert np.isnan(table["epistemic_std"][0])  # the caller's table is left unchanged


def test_write_results_fits_warns_when_values_are_missing(tmp_path):
    table = _table()
    table.meta.clear()
    table["epistemic_std"] = MaskedColumn([0.0], mask=[True])

    with pytest.warns(UserWarning, match="FITS cannot mark a float as missing"):
        write_results(table, tmp_path / "predictions.fits")


def test_write_results_fits_does_not_announce_hierarch_cards(tmp_path):
    table = _table()
    table.meta.update({"n_members": 5, "model_revision": "abc", "turbulens_version": "0.1.0"})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_results(table, tmp_path / "predictions.fits")

    assert not [record for record in caught if "HIERARCH" in str(record.message)]
    assert any("does not preserve table metadata" in str(record.message) for record in caught)
