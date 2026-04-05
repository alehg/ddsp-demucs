#!/usr/bin/env python3
"""Minimal training smoke test for DDSP model on TFRecords."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml


def _install_crepe_stub() -> None:
    """Stub crepe to allow DDSP imports in local env."""
    if "crepe" in sys.modules:
        return
    mod = types.ModuleType("crepe")

    def _predict_stub(*_args, **_kwargs):
        raise RuntimeError("crepe.predict unavailable in smoke test environment")

    mod.predict = _predict_stub
    sys.modules["crepe"] = mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a tiny DDSP training smoke test")
    p.add_argument("--config", type=Path, default=Path("env/config.yaml"))
    p.add_argument("--tfrecords-dir", type=Path, default=None)
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--frame-rate", type=int, default=250)
    p.add_argument("--win-s", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--train-steps", type=int, default=2)
    p.add_argument("--val-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--output-dir", type=Path, default=Path("results/smoke_train"))
    return p.parse_args()


def _fix_len(v: np.ndarray, n_frames: int) -> np.ndarray:
    if len(v) >= n_frames:
        return v[:n_frames].astype(np.float32)
    out = np.zeros((n_frames,), dtype=np.float32)
    out[: len(v)] = v.astype(np.float32)
    return out


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    tfrecords_dir = (args.tfrecords_dir or Path(cfg["paths"]["tfrecords_dir"])).expanduser()
    features_dir = (args.features_dir or Path(cfg["paths"]["features_dir"])).expanduser()
    out_dir = args.output_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    _install_crepe_stub()
    from ddsp_demucs.data import parse_tfrecord_example
    from ddsp_demucs.model import DDSPDecoder
    from ddsp_demucs.train import DDSPTrainer

    n_frames = int(round(args.win_s * args.frame_rate))
    feature_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    train_files = sorted((tfrecords_dir / "train").glob("*.tfrecord"))
    val_files = sorted((tfrecords_dir / "val").glob("*.tfrecord"))
    if not train_files or not val_files:
        raise RuntimeError(f"Missing TFRecords under {tfrecords_dir}/train or /val")

    def _load_features_np(track_b, start_s, end_s) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(track_b, "numpy"):
            track = track_b.numpy().decode("utf-8")
        else:
            track = bytes(track_b).decode("utf-8")
        start_v = float(start_s.numpy()) if hasattr(start_s, "numpy") else float(start_s)
        end_v = float(end_s.numpy()) if hasattr(end_s, "numpy") else float(end_s)
        if track not in feature_cache:
            npz_path = features_dir / f"{track}.features.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing feature file: {npz_path}")
            data = np.load(npz_path)
            f0_all = data["f0_hz"].astype(np.float32)
            ld_all = data["loudness_db"].astype(np.float32)
            feature_cache[track] = (f0_all, ld_all)

        f0_all, ld_all = feature_cache[track]
        fa = int(round(start_v * args.frame_rate))
        fb = int(round(end_v * args.frame_rate))
        if fb <= fa:
            fb = fa + n_frames

        f0 = _fix_len(f0_all[max(0, fa):max(0, fb)], n_frames)
        ld = _fix_len(ld_all[max(0, fa):max(0, fb)], n_frames)
        return f0, ld

    def _map_to_cond(ex: dict) -> tuple[dict, tf.Tensor]:
        xin = tf.cast(ex["audio_input"], tf.float32)
        y = tf.cast(ex["audio_target"], tf.float32)

        f0, ld = tf.py_function(
            func=_load_features_np,
            inp=[ex["track"], ex["start_sec"], ex["end_sec"]],
            Tout=[tf.float32, tf.float32],
        )
        f0.set_shape([n_frames])
        ld.set_shape([n_frames])

        cond = {
            "f0_hz": f0,
            "loudness_db": ld,
            "x_in": xin,
        }
        return cond, y

    ds_train = (
        tf.data.TFRecordDataset([str(f) for f in train_files[:2]])
        .map(parse_tfrecord_example, num_parallel_calls=1)
        .map(_map_to_cond, num_parallel_calls=1)
        .batch(args.batch_size, drop_remainder=True)
        .prefetch(1)
    )
    ds_val = (
        tf.data.TFRecordDataset([str(f) for f in val_files[:1]])
        .map(parse_tfrecord_example, num_parallel_calls=1)
        .map(_map_to_cond, num_parallel_calls=1)
        .batch(args.batch_size, drop_remainder=True)
        .prefetch(1)
    )

    model = DDSPDecoder(sample_rate=args.sample_rate, frame_rate=args.frame_rate)
    trainer = DDSPTrainer(
        model=model,
        sample_rate=args.sample_rate,
        mel_weight=1.0,
        hf_mel_weight=0.5,
        centroid_weight=0.05,
        transient_weight=0.2,
        sisdr_weight=0.0,
    )
    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        run_eagerly=True,  # ddsp.FilteredNoise expects concrete batch size in this setup.
    )

    history = trainer.fit(
        ds_train.take(args.train_steps),
        validation_data=ds_val.take(args.val_steps),
        epochs=1,
        steps_per_epoch=args.train_steps,
        validation_steps=args.val_steps,
        verbose=1,
    )

    hist = {k: [float(vv) for vv in v] for k, v in history.history.items()}
    out_json = out_dir / "train_smoke_history.json"
    out_json.write_text(json.dumps(hist, indent=2))
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()

