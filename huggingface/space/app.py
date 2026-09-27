"""Gradio Space for turbulens: predict fractal-turbulence parameters from a FITS cube or image."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download

from turbulens.inference import Inferencer
from turbulens.io.image_loaders import load_input
from turbulens.models.registry import EnsembleRegistry
from turbulens.spectral import SpectralEstimator

# Fill in with the actual HF model repo id before deploying, e.g. "yourname/turbulens-ensembles".
MODEL_REPO_ID = "<your-hf-username>/turbulens-ensembles"

_ENSEMBLE_ROOT: Path | None = None


def _ensemble_root() -> Path:
    """Download (once) and cache the model repo; return its local root."""
    global _ENSEMBLE_ROOT
    if _ENSEMBLE_ROOT is None:
        _ENSEMBLE_ROOT = Path(snapshot_download(repo_id=MODEL_REPO_ID))
    return _ENSEMBLE_ROOT


def _project(cube, channel: int | None, integrate: str):
    """Reduce a loaded Cube to a single 2D image per the UI's channel controls."""
    if cube.n_channels == 1:
        return cube
    if channel is not None:
        return cube.slice_channel(int(channel))
    if integrate.strip():
        low_str, high_str = integrate.split(":")
        return cube.project(int(low_str), int(high_str))
    return cube.project(0, cube.n_channels)


def _plot_image(image: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(image, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Input image (post-projection)")
    fig.tight_layout()
    return fig


def predict(
    file, method: str, target: str, real_data: bool, channel: str, integrate: str,
):
    if file is None:
        raise gr.Error("Upload a FITS cube, or a PNG/TIFF/.npy image first.")

    try:
        cube = load_input(file.name)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise gr.Error(f"Could not read input: {exc}") from exc

    channel_index = int(channel) if channel.strip() else None
    try:
        image_cube = _project(cube, channel_index, integrate)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    fig = _plot_image(image_cube.data)

    if method in ("fft-band", "fft-powerlaw"):
        spectral_method = "band_edge" if method == "fft-band" else "power_law"
        table = SpectralEstimator(method=spectral_method).predict(image_cube.data)
    else:
        target_arg = None if target == "multitask (all four)" else target
        try:
            members_dir = EnsembleRegistry.resolve(_ensemble_root(), target=target_arg)
            table = Inferencer(members_dir).predict(image_cube.data, real_data=real_data)
        except FileNotFoundError as exc:
            raise gr.Error(f"Model loading failed: {exc}") from exc

    df = table.to_pandas()
    return df, fig


EXAMPLES_DIR = Path(__file__).parent / "examples"
example_files = sorted(EXAMPLES_DIR.glob("*.fits*")) if EXAMPLES_DIR.is_dir() else []

with gr.Blocks(title="turbulens") as demo:
    gr.Markdown(
        "# turbulens\n"
        "Predict fractal-turbulence parameters (`k_min`, `k_max`, `sigma`, `beta`) from a "
        "radio-astronomy FITS cube or a synthetic pyFC image. "
        "[Source](https://github.com/wbandabarragan/cnn-turbulence) · "
        f"[Model weights](https://huggingface.co/{MODEL_REPO_ID})"
    )
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Input file (FITS/.fits.gz, PNG, TIFF, or .npy)")
            if example_files:
                gr.Examples(examples=[[str(p)] for p in example_files], inputs=[file_input])
            method = gr.Radio(
                ["cnn", "fft-band", "fft-powerlaw"], value="cnn", label="Method",
                info="cnn: trained ensemble. fft-band: synthetic band-limited images only. "
                "fft-powerlaw: real observational data with a continuous spectrum.",
            )
            target = gr.Dropdown(
                ["multitask (all four)", "k_min", "k_max", "sigma", "beta"],
                value="multitask (all four)", label="Target (cnn method only)",
            )
            real_data = gr.Checkbox(
                label="Real observation (--real-data)", value=False,
                info="Required for real telescope data; leave off for synthetic pyFC images. "
                "Recenters onto the training-set scale instead of the image's own pixel range.",
            )
            with gr.Row():
                channel = gr.Textbox(label="Channel (0-indexed, single-channel input)", placeholder="e.g. 5")
                integrate = gr.Textbox(label="Integrate LOW:HIGH", placeholder="e.g. 0:100")
            run_btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            output_table = gr.Dataframe(label="Predictions")
            output_plot = gr.Plot(label="Input image")

    run_btn.click(
        predict,
        inputs=[file_input, method, target, real_data, channel, integrate],
        outputs=[output_table, output_plot],
    )

if __name__ == "__main__":
    demo.launch()
