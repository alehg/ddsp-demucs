#!/usr/bin/env python3
"""Export a few val-window WAVs for listening: x_in (Demucs), y_pred (model), y_true (GT).

Uses the same TFRecord + feature NPZ pipeline as `evaluate_experiment_summary.py`.

Example:
  python scripts/export_eval_audio_samples.py \\
    --exp-name expC_residual_cold_dry0 \\
    --config configs/exp001_dualfir_tuned.yaml \\
    --env-config env/config.yaml \\
    --model-kind residual \\
    --num-samples 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
import yaml

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf


def _install_crepe_stub() -> None:
    if "crepe" in sys.modules:
        return
    mod = types.ModuleType("crepe")

    def _predict_stub(*_args, **_kwargs):
        raise RuntimeError("crepe.predict unavailable in this environment")

    mod.predict = _predict_stub
    sys.modules["crepe"] = mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export x_in / y_pred / y_true WAVs from val TFRecords")
    p.add_argument("--exp-name", type=str, required=True, help="Experiment folder under results/")
    p.add_argument("--config", type=Path, default=Path("configs/exp001_dualfir_tuned.yaml"))
    p.add_argument("--env-config", type=Path, default=Path("env/config.yaml"))
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--model-kind", type=str, default="direct", choices=["direct", "residual"])
    p.add_argument("--weights-path", type=Path, default=None, help="Defaults to results/<exp>/checkpoints/ddsp.best.weights.h5")
    p.add_argument("--val-steps", type=int, default=0, help="Override val steps; 0 = use run_config.json")
    p.add_argument("--num-samples", type=int, default=8, help="Number of 4s windows to export")
    p.add_argument(
        "--skip-samples",
        type=int,
        default=0,
        help="Skip this many val windows first (sequential order) for variety",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/<exp>/audio_eval_samples/)",
    )
    return p.parse_args()


def _fix_len(v: np.ndarray, n_frames: int) -> np.ndarray:
    if len(v) >= n_frames:
        return v[:n_frames].astype(np.float32)
    out = np.zeros((n_frames,), dtype=np.float32)
    out[: len(v)] = v.astype(np.float32)
    return out


def _safe_name(s: str, max_len: int = 64) -> str:
    out = "".join(c if c.isalnum() or c in "-._" else "_" for c in s)
    return out[:max_len] or "track"


def main() -> None:
    args = parse_args()
    _install_crepe_stub()

    from ddsp_demucs.data import parse_tfrecord_example
    from ddsp_demucs.evaluate import si_sdr
    from ddsp_demucs.model import DDSPDecoder, ResidualDDSPDecoder

    cfg = yaml.safe_load(args.config.read_text())
    env_cfg = yaml.safe_load(args.env_config.read_text())

    exp_dir = args.output_root.expanduser() / args.exp_name
    run_cfg_path = exp_dir / "run_config.json"
    run_cfg: Dict = json.loads(run_cfg_path.read_text()) if run_cfg_path.exists() else {}

    weights_path = args.weights_path or (exp_dir / "checkpoints" / "ddsp.best.weights.h5")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weights_path}")

    sample_rate = int(cfg.get("training", {}).get("sample_rate", 16000))
    frame_rate = int(cfg.get("training", {}).get("frame_rate", 250))
    win_s = float(cfg.get("training", {}).get("win_s", 4.0))
    n_frames = int(round(win_s * frame_rate))
    model_cfg = cfg.get("model", {})

    batch_size = int(run_cfg.get("batch_size", cfg.get("training", {}).get("batch_size", 8)))
    val_steps_default = int(run_cfg.get("val_steps", 0))
    val_steps = int(args.val_steps) if args.val_steps > 0 else val_steps_default
    if val_steps <= 0:
        raise RuntimeError("val_steps not in run_config.json; pass --val-steps")

    features_dir = Path(env_cfg["paths"]["features_dir"]).expanduser()
    tfrecords_dir = Path(env_cfg["paths"]["tfrecords_dir"]).expanduser()
    val_files = sorted((tfrecords_dir / "val").glob("*.tfrecord"))
    if not val_files:
        raise RuntimeError(f"No TFRecords under {tfrecords_dir}/val")

    out_dir = args.out_dir.expanduser() if args.out_dir else (exp_dir / "audio_eval_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _load_features_np(track_b, start_s, end_s):
        track = track_b.numpy().decode("utf-8") if hasattr(track_b, "numpy") else bytes(track_b).decode("utf-8")
        start_v = float(start_s.numpy()) if hasattr(start_s, "numpy") else float(start_s)
        end_v = float(end_s.numpy()) if hasattr(end_s, "numpy") else float(end_s)

        if track not in feature_cache:
            npz_path = features_dir / f"{track}.features.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing feature file: {npz_path}")
            d = np.load(npz_path)
            feature_cache[track] = (d["f0_hz"].astype(np.float32), d["loudness_db"].astype(np.float32))

        f0_all, ld_all = feature_cache[track]
        fa = int(round(start_v * frame_rate))
        fb = int(round(end_v * frame_rate))
        if fb <= fa:
            fb = fa + n_frames
        f0 = _fix_len(f0_all[max(0, fa) : max(0, fb)], n_frames)
        ld = _fix_len(ld_all[max(0, fa) : max(0, fb)], n_frames)
        return f0, ld

    def _map_to_cond(ex: dict):
        xin = tf.cast(ex["audio_input"], tf.float32)
        y = tf.cast(ex["audio_target"], tf.float32)
        f0, ld = tf.py_function(
            func=_load_features_np,
            inp=[ex["track"], ex["start_sec"], ex["end_sec"]],
            Tout=[tf.float32, tf.float32],
        )
        f0.set_shape([n_frames])
        ld.set_shape([n_frames])
        cond = {"f0_hz": f0, "loudness_db": ld, "x_in": xin}
        return cond, y, ex["track"], ex["start_sec"], ex["end_sec"]

    ds_val = (
        tf.data.TFRecordDataset([str(f) for f in val_files], num_parallel_reads=tf.data.AUTOTUNE)
        .map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_map_to_cond, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .repeat()
        .take(val_steps)
    )

    model_cls = DDSPDecoder if args.model_kind == "direct" else ResidualDDSPDecoder
    model_kw = dict(
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        n_harmonics=int(model_cfg.get("n_harmonics", 64)),
        n_noise_bands=int(model_cfg.get("n_noise_bands", 65)),
        rnn_units=int(model_cfg.get("rnn_units", 256)),
        mlp_units=tuple(model_cfg.get("mlp_units", [256, 128])),
        f0_midi_range=tuple(model_cfg.get("f0_midi_range", [24.0, 84.0])),
        z_dims=int(run_cfg.get("z_dims", model_cfg.get("z_dims", 32))),
        z_time_steps=int(run_cfg.get("z_time_steps", model_cfg.get("z_time_steps", frame_rate))),
        z_rnn_channels=int(run_cfg.get("z_rnn_channels", model_cfg.get("z_rnn_channels", 512))),
        harmonic_base_gain=float(run_cfg.get("harmonic_base_gain", 0.5)),
        harmonic_voiced_gain=float(run_cfg.get("harmonic_voiced_gain", 0.5)),
        noise_base_gain=float(run_cfg.get("noise_base_gain", 0.6)),
        noise_unvoiced_gain=float(run_cfg.get("noise_unvoiced_gain", 0.8)),
        noise_transient_gain=float(run_cfg.get("noise_transient_gain", 0.8)),
        use_learned_output_gate=bool(run_cfg.get("use_learned_output_gate", False)),
        consonant_noise_enhancements=bool(run_cfg.get("consonant_noise_enhancements", False)),
        voiced_transient_noise_leak=float(run_cfg.get("voiced_transient_noise_leak", 0.32)),
        use_noise_ducking=bool(run_cfg.get("use_noise_ducking", False)),
    )
    if model_cls is ResidualDDSPDecoder:
        model_kw["dry_logit_init"] = float(run_cfg.get("dry_logit_init", -6.0))

    model = model_cls(**model_kw)
    first = next(iter(ds_val.take(1)))
    _ = model(first[0], training=False)
    model.load_weights(str(weights_path))

    manifest_path = out_dir / "manifest.csv"
    skip_remaining = int(args.skip_samples)
    saved = 0

    with manifest_path.open("w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(
            [
                "file_idx",
                "track",
                "start_sec",
                "end_sec",
                "si_sdr_y_true_vs_y_pred",
                "si_sdr_y_true_vs_x_in",
                "delta_si_sdr_pred_minus_baseline",
            ]
        )

        for cond, target, track_b, start_sec, end_sec in ds_val:
            pred = model(cond, training=False)
            bsz = int(target.shape[0])
            for i in range(bsz):
                if skip_remaining > 0:
                    skip_remaining -= 1
                    continue
                if saved >= int(args.num_samples):
                    break

                y = target[i].numpy().astype(np.float32)
                p = pred[i].numpy().astype(np.float32)
                xin = cond["x_in"][i].numpy().astype(np.float32)
                n = min(len(y), len(p), len(xin))
                y = y[:n]
                p = p[:n]
                xin = xin[:n]

                sdr_pred = float(si_sdr(y, p))
                sdr_base = float(si_sdr(y, xin))

                track = (
                    track_b[i].numpy().decode("utf-8")
                    if hasattr(track_b[i], "numpy")
                    else bytes(track_b[i]).decode("utf-8")
                )
                st = float(start_sec[i].numpy())
                en = float(end_sec[i].numpy())
                stem = f"{saved:03d}_{_safe_name(track)}_{st:06.2f}s"

                sf.write(str(out_dir / f"{stem}_x_in.wav"), xin, sample_rate, subtype="FLOAT")
                sf.write(str(out_dir / f"{stem}_y_pred.wav"), p, sample_rate, subtype="FLOAT")
                sf.write(str(out_dir / f"{stem}_y_true.wav"), y, sample_rate, subtype="FLOAT")

                w.writerow([saved, track, f"{st:.4f}", f"{en:.4f}", f"{sdr_pred:.4f}", f"{sdr_base:.4f}", f"{sdr_pred - sdr_base:.4f}"])
                mf.flush()

                saved += 1
                if saved >= int(args.num_samples):
                    break
            if saved >= int(args.num_samples):
                break

    if saved == 0:
        raise RuntimeError("No samples exported (check --skip-samples vs dataset size)")
    if saved < int(args.num_samples):
        print(f"Warning: only exported {saved} of {args.num_samples} requested (dataset / skip exhausted).")

    print(f"Wrote {saved} triplets under: {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Sample rate: {sample_rate} Hz mono")


if __name__ == "__main__":
    main()
