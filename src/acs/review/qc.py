"""Review-oriented QC artifact output helpers."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from acs.preproc.waterfall import StreamingWaterfallResult


def save_qc_review_bundle(
    result: StreamingWaterfallResult,
    output_dir: str | Path,
    *,
    stem: str = "qc",
) -> dict[str, Path]:
    """Save waterfall image, spectrum image, and metadata JSON."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    waterfall_path = out_dir / f"{stem}_waterfall.png"
    spectrum_path = out_dir / f"{stem}_spectrum.png"
    metadata_path = out_dir / f"{stem}_metadata.json"

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    im = ax.imshow(
        result.waterfall_db,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )
    ax.set_title("Baseline-flattened display waterfall (dB)")
    ax.set_xlabel("Coarse channel index")
    ax.set_ylabel("Decimated STFT frame")
    fig.colorbar(im, ax=ax, label="dB")
    fig.savefig(waterfall_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(np.arange(result.mean_excess_db.shape[0]), result.mean_excess_db, linewidth=1.0)
    ax.set_title("Mean excess spectrum (dB)")
    ax.set_xlabel("Coarse channel index")
    ax.set_ylabel("dB")
    ax.grid(True, alpha=0.2)
    fig.savefig(spectrum_path, dpi=150)
    plt.close(fig)

    metadata_payload = dict(result.metadata)
    metadata_payload["waterfall_shape"] = list(result.waterfall_db.shape)
    metadata_payload["mean_excess_shape"] = list(result.mean_excess_db.shape)
    metadata_payload["display_frame_start_rows"] = result.display_frame_start_rows.tolist()
    metadata_payload["baseline_power_min"] = float(np.min(result.baseline_power))
    metadata_payload["baseline_power_max"] = float(np.max(result.baseline_power))
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True))

    return {
        "waterfall_image": waterfall_path,
        "spectrum_image": spectrum_path,
        "metadata_json": metadata_path,
    }
