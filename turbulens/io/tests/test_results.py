import pytest
from astropy.table import QTable

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
