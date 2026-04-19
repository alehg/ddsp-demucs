#!/usr/bin/env python3
"""Build conditioning features (f0_hz, loudness_db) and accepted_segments.csv.

Outputs:
  - <features_dir>/<track>.features.npz
  - <features_dir>/accepted_segments.csv

Run from repository root, for example::

    python scripts/conditioning_features.py --config env/config.yaml --max-tracks 10000 --f0-backend torchcrepe

This script is intentionally standalone so it can run even if importing the
full ddsp_demucs package would pull optional dependencies not present locally.

``thesis/conditioning_features_smoke.py`` is a thin wrapper that invokes this
script for backward compatibility.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchcrepe
import yaml
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract vocal conditioning features (F0 + frame loudness) and segment table"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("env/config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=1,
        help="Number of tracks to process (default 1; use a large value for full corpus)",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="",
        help="Process only this exact track folder name",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Feature sample rate",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=250,
        help="Conditioning frame rate (frames/sec)",
    )
    parser.add_argument(
        "--win-s",
        type=float,
        default=4.0,
        help="Window length in seconds for accepted_segments",
    )
    parser.add_argument(
        "--hop-s",
        type=float,
        default=1.0,
        help="Hop size in seconds for accepted_segments",
    )
    parser.add_argument(
        "--min-rms-db",
        type=float,
        default=-50.0,
        help="Minimum RMS dB gate for accepted segments",
    )
    parser.add_argument(
        "--min-voiced-frac",
        type=float,
        default=0.2,
        help="Minimum voiced frame fraction gate",
    )
    parser.add_argument(
        "--max-polyphony-risk",
        type=float,
        default=1.0,
        help="Maximum allowed polyphony risk [0,1]; 1.0 disables this gate",
    )
    parser.add_argument(
        "--poly-iqr-st-thresh",
        type=float,
        default=3.0,
        help="Semitone IQR threshold where polyphony risk reaches 1.0",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="If >0, truncate each track for faster testing",
    )
    parser.add_argument(
        "--f0-backend",
        choices=("torchcrepe", "librosa"),
        default="torchcrepe",
        help="F0 extraction backend (use librosa for fast smoke on CPU)",
    )
    parser.add_argument(
        "--pyin-mode",
        choices=("default", "fast"),
        default="default",
        help="librosa.pyin mode when --f0-backend librosa",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute existing .features.npz files",
    )
    return parser.parse_args()


def torchcrepe_f0(audio_1d: np.ndarray, sr: int, hop_length: int, fmin: float = 50.0, fmax: float = 1100.0) -> np.ndarray:
    """Extract F0 in Hz using torchcrepe."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(audio_1d, dtype=torch.float32, device=device)[None]  # [1, T]
    with torch.no_grad():
        f0 = torchcrepe.predict(
            x,
            sr,
            hop_length,
            torch.tensor([fmin], device=device),
            torch.tensor([fmax], device=device),
            model="full",
            batch_size=1024,
            device=device,
            return_periodicity=False,
        )[0].cpu().numpy()
    return np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)


def librosa_f0(audio_1d: np.ndarray, sr: int, hop_length: int, mode: str = "default") -> np.ndarray:
    """Extract F0 in Hz using librosa.pyin."""
    kwargs = dict(
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
        hop_length=hop_length,
    )
    if mode == "fast":
        kwargs.update(
            n_thresholds=32,
            resolution=0.2,
            max_transition_rate=50.0,
        )
    f0, _, _ = librosa.pyin(audio_1d, **kwargs)
    f0 = np.asarray(f0, dtype=np.float32)
    return np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)


def rms_db(x: np.ndarray, eps: float = 1e-8) -> float:
    rms = np.sqrt(np.mean(x.astype(np.float32) ** 2) + eps)
    return float(20.0 * np.log10(rms + eps))


def frame_loudness_db(audio_1d: np.ndarray, hop_length: int, ref_db: float = 20.7) -> np.ndarray:
    """Frame-wise loudness proxy in dB from RMS energy."""
    rms = librosa.feature.rms(
        y=audio_1d,
        frame_length=2048,
        hop_length=hop_length,
        center=True,
    )[0]
    return (20.0 * np.log10(np.maximum(rms, 1e-8)) - ref_db).astype(np.float32)


def segment_polyphony_risk(f0_segment: np.ndarray, iqr_st_thresh: float = 3.0) -> float:
    """Estimate polyphony risk from voiced F0 spread in semitones.

    A monophonic sung segment tends to have a tighter F0 distribution.
    Segments with simultaneous voices/harmony often show broader voiced-F0 spread.
    """
    voiced = f0_segment[f0_segment > 0.0]
    if voiced.size < 8:
        return 0.0
    st = 12.0 * np.log2(np.maximum(voiced, 1e-6) / 55.0)
    iqr = float(np.percentile(st, 75) - np.percentile(st, 25))
    if iqr_st_thresh <= 0:
        return 0.0
    risk = iqr / iqr_st_thresh
    return float(np.clip(risk, 0.0, 1.0))


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    stems_dir = Path(cfg["paths"]["stems_dir"]).expanduser()
    features_dir = Path(cfg["paths"]["features_dir"]).expanduser()
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stems dir: {stems_dir}")
    print(f"Features dir: {features_dir}")
    print(f"Tracks to process: {args.max_tracks}")

    hop_len = int(round(args.sample_rate / args.frame_rate))
    win = int(round(args.win_s * args.sample_rate))
    hop = int(round(args.hop_s * args.sample_rate))

    all_track_dirs = sorted([p for p in stems_dir.iterdir() if p.is_dir()])
    if args.track:
        track_dirs = [p for p in all_track_dirs if p.name == args.track]
        if not track_dirs:
            raise RuntimeError(f"Track not found under stems_dir: {args.track}")
    else:
        track_dirs = all_track_dirs[: args.max_tracks]
    if not track_dirs:
        raise RuntimeError(f"No track directories found in {stems_dir}")

    rows = []
    for td in tqdm(track_dirs, desc="Conditioning features"):
        mono_wav = td / "vocals.mono.wav"
        if not mono_wav.exists():
            continue

        feature_path = features_dir / f"{td.name}.features.npz"
        if feature_path.exists() and not args.force:
            print(f"Skipping existing features: {feature_path.name}")
            continue

        y, sr = sf.read(str(mono_wav), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != args.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=args.sample_rate)
            sr = args.sample_rate

        if args.max_seconds > 0:
            y = y[: int(math.floor(args.max_seconds * args.sample_rate))]

        if y.size < win:
            print(f"Skipping short track (< win_s): {td.name}")
            continue

        # Full-track conditioning arrays
        if args.f0_backend == "torchcrepe":
            f0 = torchcrepe_f0(y, sr=sr, hop_length=hop_len)
        else:
            f0 = librosa_f0(y, sr=sr, hop_length=hop_len, mode=args.pyin_mode)
        ld = frame_loudness_db(y.astype(np.float32), hop_length=hop_len, ref_db=20.7)

        np.savez_compressed(
            feature_path,
            f0_hz=f0,
            loudness_db=ld,
            sample_rate=np.int32(args.sample_rate),
            frame_rate=np.int32(args.frame_rate),
        )

        # Segment windows for accepted_segments.csv
        N = len(y)
        for a in range(0, N - win + 1, hop):
            b = a + win
            seg = y[a:b]
            seg_rms_db = rms_db(seg)

            fa = int(round((a / args.sample_rate) * args.frame_rate))
            fb = int(round((b / args.sample_rate) * args.frame_rate))
            fseg = f0[fa:fb] if fb > fa else np.array([], dtype=np.float32)
            voiced_frac = float(np.mean(fseg > 0.0)) if fseg.size else 0.0
            polyphony_risk = segment_polyphony_risk(fseg, iqr_st_thresh=args.poly_iqr_st_thresh) if fseg.size else 0.0
            mono_fraction = 1.0 - polyphony_risk

            segment_pass = (
                (seg_rms_db >= args.min_rms_db)
                and (voiced_frac >= args.min_voiced_frac)
                and (polyphony_risk <= args.max_polyphony_risk)
            )
            rows.append(
                {
                    "track": td.name,
                    "start_s": a / args.sample_rate,
                    "end_s": b / args.sample_rate,
                    "rms_db": seg_rms_db,
                    "voiced_frac": voiced_frac,
                    "polyphony_risk": polyphony_risk,
                    "mono_fraction": mono_fraction,
                    "segment_pass": bool(segment_pass),
                }
            )

    if not rows:
        raise RuntimeError("No segments produced; check inputs and parameters.")

    accepted = pd.DataFrame(rows)
    accepted_path = features_dir / "accepted_segments.csv"
    accepted.to_csv(accepted_path, index=False)
    print(f"Wrote: {accepted_path}")
    print(f"Total segments: {len(accepted)} | accepted: {int(accepted['segment_pass'].sum())}")


if __name__ == "__main__":
    main()
