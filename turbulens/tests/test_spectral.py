import numpy as np
import pytest

from turbulens.spectral import SpectralEstimator, estimate_band_edges, fit_inertial_range


def _power_law_image(n=256, alpha=-3.67, seed=0):
    """Synthesize a real image whose radial power spectrum follows P(k) ~ k^alpha."""
    rng = np.random.default_rng(seed)
    kax = np.fft.fftshift(np.fft.fftfreq(n)) * n
    kxg, kyg = np.meshgrid(kax, kax)
    kr = np.sqrt(kxg**2 + kyg**2)
    amp = np.clip(kr, 1.0, None) ** (alpha / 2.0)
    phase = np.exp(2j * np.pi * rng.random((n, n)))
    spectrum = np.fft.ifftshift(amp * phase)
    return np.fft.ifft2(spectrum).real


def _band_limited_image(n=256, kmin=10, kmax=60, seed=0):
    """Synthesize a real image whose power lives only in [kmin, kmax]."""
    rng = np.random.default_rng(seed)
    kax = np.fft.fftshift(np.fft.fftfreq(n)) * n
    kxg, kyg = np.meshgrid(kax, kax)
    kr = np.sqrt(kxg**2 + kyg**2)
    amp = np.where((kr >= kmin) & (kr <= kmax), np.clip(kr, 1e-6, None) ** (-1.6667 - 2.0), 0.0)
    phase = np.exp(2j * np.pi * rng.random((n, n)))
    spectrum = np.fft.ifftshift(amp * phase)
    return np.fft.ifft2(spectrum).real


def test_estimate_band_edges_recovers_kmin_kmax():
    image = _band_limited_image(kmin=10, kmax=60)

    edges = estimate_band_edges(image)

    assert abs(edges["k_min"] - 10) <= 3
    assert abs(edges["k_max"] - 60) <= 3


def test_estimate_band_edges_returns_nan_for_constant_image():
    edges = estimate_band_edges(np.zeros((32, 32)))

    assert np.isnan(edges["k_min"])
    assert np.isnan(edges["k_max"])


def test_estimate_band_edges_returns_nan_for_white_noise():
    # White noise has no real spectral edge; the adaptive noise-floor threshold should
    # recognize this (rather than reporting some arbitrary low/high k as a "band").
    rng = np.random.default_rng(0)
    edges = estimate_band_edges(rng.normal(size=(64, 64)))

    assert np.isnan(edges["k_min"])
    assert np.isnan(edges["k_max"])


def test_estimate_band_edges_warns_when_result_hits_grid_boundary():
    # A continuous power-law spectrum (no sharp band edge at all) run through the
    # band-edge estimator it is not meant for: it never finds a real edge, so its
    # answer is the FFT grid boundary, and that must be surfaced, not silent.
    with pytest.warns(UserWarning, match="edge of the available wavenumber grid"):
        estimate_band_edges(_power_law_image(alpha=-3.67))


def test_estimate_band_edges_noise_floor_rejects_quantization_noise():
    # A band-limited image whose out-of-band bins carry small but nonzero
    # quantization noise (as an 8-bit PNG round-trip would introduce) must not
    # have its adaptively-set threshold fooled into reporting the FFT grid edge.
    image = _band_limited_image(kmin=10, kmax=60)
    rng = np.random.default_rng(1)
    quantized = np.round((image - image.min()) / (image.max() - image.min()) * 255) / 255.0
    quantized = quantized + rng.normal(scale=1e-4, size=quantized.shape)

    edges = estimate_band_edges(quantized)

    assert abs(edges["k_max"] - 60) <= 5


def test_spectral_estimator_predict_returns_kmin_kmax_rows():
    image = _band_limited_image(kmin=10, kmax=60)

    table = SpectralEstimator().predict(image)

    assert list(table["target"]) == ["k_min", "k_max"]
    assert list(table["image_index"]) == [0, 0]
    assert abs(float(table[0]["value"]) - 10) <= 3
    assert abs(float(table[1]["value"]) - 60) <= 3
    assert np.isnan(table[0]["epistemic_std"])


def test_spectral_estimator_predict_accepts_a_batch_of_images():
    images = np.stack([_band_limited_image(kmin=10, kmax=60, seed=i) for i in range(3)])

    table = SpectralEstimator().predict(images)

    assert len(table) == 6  # 3 images x 2 targets
    assert list(table["image_index"]) == [0, 0, 1, 1, 2, 2]


def test_spectral_estimator_predict_rejects_invalid_image_shape():
    with pytest.raises(ValueError):
        SpectralEstimator().predict(np.zeros((2, 3, 64, 64)))


def test_fit_inertial_range_recovers_spectral_index():
    image = _power_law_image(alpha=-3.67)

    fit = fit_inertial_range(image)

    assert fit["alpha"] == pytest.approx(-3.67, abs=0.3)
    assert fit["k_min"] < fit["k_max"]


def test_fit_inertial_range_returns_nan_for_constant_image():
    fit = fit_inertial_range(np.zeros((32, 32)))

    assert np.isnan(fit["alpha"])
    assert np.isnan(fit["r_squared"])
    assert np.isnan(fit["k_min"])
    assert np.isnan(fit["k_max"])


def test_fit_inertial_range_r_squared_distinguishes_real_power_law_from_noise():
    rng = np.random.default_rng(0)
    power_law_fit = fit_inertial_range(_power_law_image(alpha=-3.67))
    noise_fit = fit_inertial_range(rng.normal(size=(256, 256)))

    assert power_law_fit["r_squared"] > 0.8
    assert noise_fit["r_squared"] < 0.5


def test_fit_inertial_range_warns_when_no_departure_found():
    # A threshold far tighter than the fit's own residual noise means no point,
    # not even inside the fit range, is called "in range" -- k_min/k_max then
    # fall back to the fit range's own (unmeasured) edges, which must be
    # surfaced, not returned silently as if measured.
    image = _power_law_image(alpha=-3.67)

    with pytest.warns(UserWarning, match="no point in the fit range departs"):
        fit = fit_inertial_range(image, threshold=1e-6)

    assert fit["k_min"] == 6.0  # k_lo fallback
    assert fit["k_max"] == 89.0  # k_hi fallback


def test_spectral_estimator_rejects_unknown_method():
    with pytest.raises(ValueError):
        SpectralEstimator(method="bogus")


def test_spectral_estimator_power_law_predict_returns_four_rows_including_alpha_and_r_squared():
    image = _power_law_image(alpha=-3.67)

    table = SpectralEstimator(method="power_law").predict(image)

    assert list(table["target"]) == ["k_min", "k_max", "alpha", "r_squared"]
    assert float(table[0]["value"]) < float(table[1]["value"])
    assert np.isnan(table[0]["epistemic_std"])
    values = dict(zip(table["target"], table["value"]))
    assert values["alpha"] == pytest.approx(-3.67, abs=0.3)
    assert values["r_squared"] > 0.8
