#!/usr/bin/env python3
"""Run baseline reference experiment: Demucs vocals vs MUSDB target vocals."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import librosa
import musdb
import numpy as np
import pandas as pd
import soundfile as sf
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Demucs baseline on test split segments")
    parser.add_argument("--config", type=Path, default=Path("env/config.yaml"), help="Path to YAML config")
    parser.add_argument(
        "--segment-splits-csv",
        type=Path,
        default=Path("data/tfrecords/split_metadata/segment_splits.csv"),
        help="Path to segment split CSV produced during TFRecord generation",
    )
    parser.add_argument("--stems-dir", type=Path, default=None, help="Path to Demucs stems directory")
    parser.add_argument("--musdb-root", type=Path, default=None, help="Path to MUSDB root")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Evaluation sample rate")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split to evaluate")
    parser.add_argument("--max-segments", type=int, default=0, help="If >0, evaluate only this many segments")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for segment subsampling")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exp000_demucs_baseline"),
        help="Output directory for metrics and summary",
    )
    return parser.parse_args()


def _load_evaluate_module():
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "src" / "ddsp_demucs" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("ddsp_evaluate_local", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluate module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_mono_resampled(wav_path: Path, target_sr: int) -> np.ndarray:
    y, sr = sf.read(str(wav_path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32)


def slice_sec(x: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    a = int(round(start_s * sr))
    b = int(round(end_s * sr))
    a = max(0, min(a, len(x)))
    b = max(0, min(b, len(x)))
    if b <= a:
        return np.zeros(1, dtype=np.float32)
    return x[a:b]


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    eval_mod = _load_evaluate_module()

    stems_dir = (args.stems_dir or Path(cfg["paths"]["stems_dir"])).expanduser()
    musdb_root = (args.musdb_root or Path(cfg["dataset"]["root"])).expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    segs = pd.read_csv(args.segment_splits_csv.expanduser())
    segs = segs[segs["split"] == args.split].copy()
    if segs.empty:
        raise RuntimeError(f"No segments found for split={args.split}")

    if args.max_segments > 0 and len(segs) > args.max_segments:
        segs = segs.sample(n=args.max_segments, random_state=args.seed).sort_values(["track", "start_s"]).reset_index(drop=True)

    db = musdb.DB(root=str(musdb_root), subsets=["train", "test"], is_wav=True)
    name2track = {t.name: t for t in db.tracks}

    rows: list[dict] = []

    for track_name, group in segs.groupby("track"):
        stem_wav = stems_dir / track_name / "vocals.mono.wav"
        if not stem_wav.exists():
            continue
        mt = name2track.get(track_name)
        if mt is None:
            continue

        pred = load_mono_resampled(stem_wav, target_sr=args.sample_rate)
        gt = mt.targets["vocals"].audio
        if gt.ndim == 2:
            gt = gt.mean(axis=1)
        gt = gt.astype(np.float32)
        if int(mt.rate) != args.sample_rate:
            gt = librosa.resample(gt, orig_sr=int(mt.rate), target_sr=args.sample_rate)

        for row in group.itertuples(index=False):
            start_s = float(row.start_s)
            end_s = float(row.end_s)
            seg_pred = slice_sec(pred, args.sample_rate, start_s, end_s)
            seg_gt = slice_sec(gt, args.sample_rate, start_s, end_s)
            n = min(len(seg_pred), len(seg_gt))
            if n < 32:
                continue
            seg_pred = seg_pred[:n]
            seg_gt = seg_gt[:n]

            si = float(eval_mod.si_sdr(seg_gt, seg_pred))
            sc = float(eval_mod.spectral_convergence(seg_gt, seg_pred))
            rows.append(
                {
                    "track": track_name,
                    "start_s": start_s,
                    "end_s": end_s,
                    "dur_s": end_s - start_s,
                    "si_sdr": si,
                    "spectral_convergence": sc,
                }
            )

    if not rows:
        raise RuntimeError("No evaluation rows were produced")

    metrics_df = pd.DataFrame(rows)
    metrics_csv = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    summary = {
        "split": args.split,
        "sample_rate": args.sample_rate,
        "n_segments": int(len(metrics_df)),
        "n_tracks": int(metrics_df["track"].nunique()),
        "si_sdr_mean": float(metrics_df["si_sdr"].mean()),
        "si_sdr_median": float(metrics_df["si_sdr"].median()),
        "si_sdr_std": float(metrics_df["si_sdr"].std(ddof=0)),
        "spectral_convergence_mean": float(metrics_df["spectral_convergence"].mean()),
        "spectral_convergence_median": float(metrics_df["spectral_convergence"].median()),
        "spectral_convergence_std": float(metrics_df["spectral_convergence"].std(ddof=0)),
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {summary_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

