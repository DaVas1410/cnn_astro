import json

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.table import QTable

from turbulens.cli import main
from turbulens.models.architecture import build_model_from_config


def _write_fits_cube(path):
    data = np.zeros((4, 64, 64), dtype=np.float32)
    fits.PrimaryHDU(data=data).writeto(path)


def _write_member(member_dir):
    member_dir.mkdir(parents=True)
    config = {
        "model": {
            "architecture": "resnet18", "in_channels": 1, "pretrained": False,
            "dropout": 0.1, "input_standardization": "none",
        },
        "task": {"targets": ["k_min"], "target_ranges": {"k_min": [1.0, 32.0]}},
    }
    (member_dir / "configuration.json").write_text(json.dumps(config))
    model = build_model_from_config(config)
    checkpoint_dir = member_dir / "checkpoints"
    checkpoint_dir.mkdir()
    payload = {
        "model_state": model.state_dict(),
        "normalization_stats": {
            "lower_percentile": 0.5, "upper_percentile": 99.5,
            "lower_value": -1.0, "upper_value": 3.0, "value_range": 4.0,
            "finite_pixels_examined": 1000, "images_examined": 10,
            "sampling_seed": 1, "source": "test_fixture",
        },
    }
    torch.save(payload, checkpoint_dir / "best_checkpoint.pt")
    (member_dir / "COMPLETED.json").write_text("{}")


def test_infer_command_writes_results_with_explicit_model_path(tmp_path):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)
    model_root = tmp_path / "ensemble_root"
    _write_member(model_root / "member_0")
    _write_member(model_root / "member_1")
    out_path = tmp_path / "predictions.ecsv"

    main([
        "infer",
        "--input", str(fits_path),
        "--channel", "0",
        "--model", str(model_root),
        "--out", str(out_path),
    ])

    table = QTable.read(out_path)
    assert list(table["target"]) == ["k_min"]
    assert table.meta["input_path"] == str(fits_path)
    assert table.meta["turbulens_version"]


def test_infer_command_resolves_model_auto_from_output_root(tmp_path):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)
    output_root = tmp_path / "pipeline_output"
    _write_member(output_root / "multitask" / "v1" / "ensemble" / "members" / "member_0")
    _write_member(output_root / "multitask" / "v1" / "ensemble" / "members" / "member_1")
    out_path = tmp_path / "predictions.ecsv"

    main([
        "infer",
        "--input", str(fits_path),
        "--integrate", "0:2",
        "--model", "auto",
        "--output-root", str(output_root),
        "--out", str(out_path),
    ])

    table = QTable.read(out_path)
    assert len(table) == 1


def test_infer_command_resolves_model_auto_with_target_to_single_target_ensemble(tmp_path):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)
    output_root = tmp_path / "pipeline_output"
    _write_member(output_root / "single_k_min" / "v1" / "ensemble" / "members" / "member_0")
    _write_member(output_root / "single_k_min" / "v1" / "ensemble" / "members" / "member_1")
    out_path = tmp_path / "predictions.ecsv"

    main([
        "infer",
        "--input", str(fits_path),
        "--integrate", "0:2",
        "--model", "auto",
        "--target", "k_min",
        "--output-root", str(output_root),
        "--out", str(out_path),
    ])

    table = QTable.read(out_path)
    assert list(table["target"]) == ["k_min"]


def test_infer_command_rejects_invalid_target_choice(tmp_path, capsys):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)

    with pytest.raises(SystemExit):
        main([
            "infer",
            "--input", str(fits_path),
            "--integrate", "0:2",
            "--model", "auto",
            "--target", "not_a_target",
            "--output-root", str(tmp_path),
        ])

    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


def test_infer_command_rejects_target_without_model_auto(tmp_path, capsys):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)
    model_root = tmp_path / "ensemble_root"
    _write_member(model_root / "member_0")
    _write_member(model_root / "member_1")

    with pytest.raises(SystemExit):
        main([
            "infer",
            "--input", str(fits_path),
            "--integrate", "0:2",
            "--model", str(model_root),
            "--target", "k_min",
        ])

    captured = capsys.readouterr()
    assert "usage:" in captured.err
    assert "--target" in captured.err


def test_infer_command_loads_options_from_config_json(tmp_path):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)
    output_root = tmp_path / "pipeline_output"
    _write_member(output_root / "single_k_min" / "v1" / "ensemble" / "members" / "member_0")
    _write_member(output_root / "single_k_min" / "v1" / "ensemble" / "members" / "member_1")
    out_path = tmp_path / "predictions.ecsv"
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps({
        "input": str(fits_path),
        "integrate": "0:2",
        "model": "auto",
        "target": "k_min",
        "output_root": str(output_root),
        "out": str(out_path),
    }))

    main(["infer", "--config", str(config_path)])

    table = QTable.read(out_path)
    assert list(table["target"]) == ["k_min"]


def test_infer_command_cli_flag_overrides_config_json(tmp_path):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)
    output_root = tmp_path / "pipeline_output"
    _write_member(output_root / "single_k_min" / "v1" / "ensemble" / "members" / "member_0")
    _write_member(output_root / "single_k_min" / "v1" / "ensemble" / "members" / "member_1")
    config_out_path = tmp_path / "from_config.ecsv"
    cli_out_path = tmp_path / "from_cli.ecsv"
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps({
        "input": str(fits_path),
        "integrate": "0:2",
        "model": "auto",
        "target": "k_min",
        "output_root": str(output_root),
        "out": str(config_out_path),
    }))

    main(["infer", "--config", str(config_path), "--out", str(cli_out_path)])

    assert cli_out_path.exists()
    assert not config_out_path.exists()


def test_infer_command_rejects_unknown_config_key(tmp_path, capsys):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps({"input": str(fits_path), "bogus_option": "x"}))

    with pytest.raises(SystemExit):
        main(["infer", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert "usage:" in captured.err
    assert "bogus_option" in captured.err


def test_infer_command_rejects_non_object_config_json(tmp_path, capsys):
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(["not", "an", "object"]))

    with pytest.raises(SystemExit):
        main(["infer", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert "usage:" in captured.err
    assert "JSON object" in captured.err


def test_infer_command_rejects_malformed_config_json(tmp_path, capsys):
    config_path = tmp_path / "run.json"
    config_path.write_text("{not valid json")

    with pytest.raises(SystemExit):
        main(["infer", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert "usage:" in captured.err
    assert "could not read" in captured.err


def test_infer_command_prints_to_stdout_when_out_omitted(tmp_path, capsys):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)
    model_root = tmp_path / "ensemble_root"
    _write_member(model_root / "member_0")
    _write_member(model_root / "member_1")

    main(["infer", "--input", str(fits_path), "--channel", "0", "--model", str(model_root)])

    captured = capsys.readouterr()
    assert "k_min" in captured.out


def test_infer_command_rejects_both_channel_and_integrate(tmp_path):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)
    model_root = tmp_path / "ensemble_root"
    _write_member(model_root / "member_0")
    _write_member(model_root / "member_1")

    with pytest.raises(SystemExit):
        main([
            "infer",
            "--input", str(fits_path),
            "--channel", "0",
            "--integrate", "0:2",
            "--model", str(model_root),
        ])


def test_infer_command_requires_channel_or_integrate(tmp_path):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)

    with pytest.raises(SystemExit):
        main(["infer", "--input", str(fits_path), "--model", "auto"])


def test_infer_command_rejects_malformed_integrate(tmp_path, capsys):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)
    model_root = tmp_path / "ensemble_root"
    _write_member(model_root / "member_0")
    _write_member(model_root / "member_1")

    with pytest.raises(SystemExit):
        main([
            "infer",
            "--input", str(fits_path),
            "--integrate", "5",
            "--model", str(model_root),
        ])

    captured = capsys.readouterr()
    assert "usage:" in captured.err


def test_infer_command_rejects_missing_output_root_with_usage_message(tmp_path, capsys):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)

    with pytest.raises(SystemExit):
        main([
            "infer",
            "--input", str(fits_path),
            "--channel", "0",
            "--model", "auto",
        ])

    captured = capsys.readouterr()
    assert "usage:" in captured.err
    assert "--output-root" in captured.err


def test_inspect_command_prints_breakdown_to_stdout(tmp_path, capsys):
    import numpy as np
    np.save(tmp_path / "image.npy", np.zeros((4, 4), dtype=np.float32))

    main(["inspect", "--input", str(tmp_path / "image.npy")])

    captured = capsys.readouterr()
    assert "Shape:" in captured.out
    assert "already-projected image" in captured.out.lower()


def test_inspect_command_writes_json_and_html(tmp_path):
    import json
    import numpy as np
    np.save(tmp_path / "image.npy", np.zeros((4, 4), dtype=np.float32))
    json_path = tmp_path / "out.json"
    html_path = tmp_path / "out.html"

    main([
        "inspect", "--input", str(tmp_path / "image.npy"),
        "--json", str(json_path), "--html", str(html_path),
    ])

    payload = json.loads(json_path.read_text())
    assert payload["n_channels"] == 1
    assert "<html>" in html_path.read_text()


def test_inspect_command_requires_input(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        main(["inspect"])


def _write_velocity_fits_cube(path):
    from astropy.io import fits
    header = fits.Header()
    header["CTYPE3"] = "VELO-LSR"
    header["CRVAL3"], header["CRPIX3"], header["CDELT3"], header["CUNIT3"] = -1000.0, 1.0, 500.0, "m/s"
    data = np.zeros((5, 64, 64), dtype=np.float32)
    fits.PrimaryHDU(data=data, header=header).writeto(path)


def test_infer_command_accepts_2d_input_without_channel_flags(tmp_path):
    np.save(tmp_path / "image.npy", np.zeros((64, 64), dtype=np.float32))
    model_root = tmp_path / "ensemble_root"
    _write_member(model_root / "member_0")
    _write_member(model_root / "member_1")
    out_path = tmp_path / "predictions.ecsv"

    main([
        "infer", "--input", str(tmp_path / "image.npy"),
        "--model", str(model_root), "--out", str(out_path),
    ])

    from astropy.table import QTable
    table = QTable.read(out_path)
    assert len(table) == 1


def test_infer_command_requires_a_selection_for_multichannel_input(tmp_path):
    fits_path = tmp_path / "cube.fits"
    _write_fits_cube(fits_path)
    model_root = tmp_path / "ensemble_root"
    _write_member(model_root / "member_0")
    _write_member(model_root / "member_1")

    with pytest.raises(SystemExit):
        main(["infer", "--input", str(fits_path), "--model", str(model_root)])


def test_infer_command_rejects_selection_for_single_channel_input(tmp_path):
    np.save(tmp_path / "image.npy", np.zeros((64, 64), dtype=np.float32))
    model_root = tmp_path / "ensemble_root"
    _write_member(model_root / "member_0")
    _write_member(model_root / "member_1")

    with pytest.raises(SystemExit):
        main([
            "infer", "--input", str(tmp_path / "image.npy"),
            "--channel", "0", "--model", str(model_root),
        ])


def test_infer_command_method_fft_band_needs_no_model(tmp_path):
    np.save(tmp_path / "image.npy", np.zeros((64, 64), dtype=np.float32))
    out_path = tmp_path / "predictions.ecsv"

    main([
        "infer", "--input", str(tmp_path / "image.npy"),
        "--method", "fft-band", "--out", str(out_path),
    ])

    table = QTable.read(out_path)
    assert list(table["target"]) == ["k_min", "k_max"]
    assert table.meta["method"] == "fft-band"
    assert "model" not in table.meta


def test_infer_command_method_fft_band_recovers_band_edges(tmp_path):
    n = 64
    kax = np.fft.fftshift(np.fft.fftfreq(n)) * n
    kxg, kyg = np.meshgrid(kax, kax)
    kr = np.sqrt(kxg**2 + kyg**2)
    kmin, kmax = 5, 20
    amp = np.where((kr >= kmin) & (kr <= kmax), 1.0, 0.0)
    rng = np.random.default_rng(0)
    phase = np.exp(2j * np.pi * rng.random((n, n)))
    image = np.fft.ifft2(np.fft.ifftshift(amp * phase)).real.astype(np.float32)
    np.save(tmp_path / "image.npy", image)
    out_path = tmp_path / "predictions.ecsv"

    main([
        "infer", "--input", str(tmp_path / "image.npy"),
        "--method", "fft-band", "--out", str(out_path),
    ])

    table = QTable.read(out_path)
    values = dict(zip(table["target"], table["value"]))
    assert abs(values["k_min"] - kmin) <= 3
    assert abs(values["k_max"] - kmax) <= 3


def test_infer_command_method_fft_powerlaw_needs_no_model(tmp_path):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "image.npy", rng.normal(size=(64, 64)).astype(np.float32))
    out_path = tmp_path / "predictions.ecsv"

    main([
        "infer", "--input", str(tmp_path / "image.npy"),
        "--method", "fft-powerlaw", "--out", str(out_path),
    ])

    table = QTable.read(out_path)
    assert list(table["target"]) == ["k_min", "k_max", "alpha", "r_squared"]
    assert table.meta["method"] == "fft-powerlaw"
    assert "model" not in table.meta


def test_inspect_command_reports_spectral_estimate_for_projected_image(tmp_path, capsys):
    np.save(tmp_path / "image.npy", np.zeros((4, 4), dtype=np.float32))

    main(["inspect", "--input", str(tmp_path / "image.npy")])

    captured = capsys.readouterr()
    assert "Classical FFT estimate" in captured.out


def test_infer_command_velocity_flag_end_to_end(tmp_path):
    fits_path = tmp_path / "vcube.fits"
    _write_velocity_fits_cube(fits_path)
    model_root = tmp_path / "ensemble_root"
    _write_member(model_root / "member_0")
    _write_member(model_root / "member_1")
    out_path = tmp_path / "predictions.ecsv"

    main([
        "infer", "--input", str(fits_path),
        "--velocity", "-0.5", "0.5",
        "--model", str(model_root), "--out", str(out_path),
    ])

    from astropy.table import QTable
    table = QTable.read(out_path)
    assert len(table) == 1
