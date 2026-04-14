"""CLI command for streaming waterfall/QC generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from acs.io.manifest import load_manifest
from acs.preproc.waterfall import build_streaming_waterfall
from acs.review.qc import save_qc_review_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build streaming waterfall/QC outputs")
    parser.add_argument("manifest", type=Path, help="Path to observation manifest JSON")
    parser.add_argument("output_dir", type=Path, help="Output directory for QC artifacts")
    parser.add_argument("--nfft", type=int, default=64)
    parser.add_argument("--hop", type=int, default=16)
    parser.add_argument("--block-rows", type=int, default=1024)
    parser.add_argument("--baseline-sample-frames", type=int, default=128)
    parser.add_argument("--baseline-smooth-width", type=int, default=33)
    parser.add_argument("--time-decimation", type=int, default=4)
    parser.add_argument("--freq-decimation", type=int, default=1)
    parser.add_argument("--sample-rate-hz", type=float, default=1.0)
    parser.add_argument("--start-freq-hz", type=float, default=0.0)
    parser.add_argument("--coarse-channel-spacing-hz", type=float, default=1.0)
    parser.add_argument("--stem", type=str, default="qc")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    result = build_streaming_waterfall(
        manifest,
        nfft=args.nfft,
        hop=args.hop,
        block_rows=args.block_rows,
        baseline_sample_frames=args.baseline_sample_frames,
        baseline_smooth_width=args.baseline_smooth_width,
        time_decimation=args.time_decimation,
        freq_decimation=args.freq_decimation,
        sample_rate_hz=args.sample_rate_hz,
        start_freq_hz=args.start_freq_hz,
        coarse_channel_spacing_hz=args.coarse_channel_spacing_hz,
    )
    paths = save_qc_review_bundle(result, args.output_dir, stem=args.stem)

    print(f"Saved waterfall image: {paths['waterfall_image']}")
    print(f"Saved spectrum image: {paths['spectrum_image']}")
    print(f"Saved metadata JSON: {paths['metadata_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
