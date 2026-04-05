#!/usr/bin/env python3
"""Train Experiment 001: DDSP direct model on TFRecords."""

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
    """Stub crepe so local DDSP imports work in this environment."""
    if "crepe" in sys.modules:
        return
    mod = types.ModuleType("crepe")

    def _predict_stub(*_args, **_kwargs):
        raise RuntimeError("crepe.predict unavailable in this environment")

    mod.predict = _predict_stub
    sys.modules["crepe"] = mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DDSP direct (exp001)")
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"), help="Base experiment config")
    p.add_argument("--env-config", type=Path, default=Path("env/config.yaml"), help="Environment/path config")
    p.add_argument("--exp-name", type=str, default="exp001_ddsp_direct")
    p.add_argument("--tfrecords-dir", type=Path, default=None)
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=Path("exp"))
    p.add_argument("--batch-size", type=int, default=0, help="Override config.training.batch_size if >0")
    p.add_argument("--epochs", type=int, default=0, help="Override config.training.epochs if >0")
    p.add_argument("--learning-rate", type=float, default=0.0, help="Override config.training.learning_rate if >0")
    p.add_argument("--train-steps", type=int, default=0, help="If >0, limit steps per epoch (smoke/debug)")
    p.add_argument("--val-steps", type=int, default=0, help="If >0, limit validation steps (smoke/debug)")
    p.add_argument("--num-shards-train", type=int, default=0, help="If >0, use only first N train shards")
    p.add_argument("--num-shards-val", type=int, default=0, help="If >0, use only first N val shards")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def _fix_len(v: np.ndarray, n_frames: int) -> np.ndarray:
    if len(v) >= n_frames:
        return v[:n_frames].astype(np.float32)
    out = np.zeros((n_frames,), dtype=np.float32)
    out[: len(v)] = v.astype(np.float32)
    return out


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)

    cfg = yaml.safe_load(args.config.read_text())
    env_cfg = yaml.safe_load(args.env_config.read_text())

    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})

    sample_rate = int(training_cfg.get("sample_rate", 16000))
    frame_rate = int(training_cfg.get("frame_rate", 250))
    win_s = float(training_cfg.get("win_s", 4.0))
    n_frames = int(round(win_s * frame_rate))

    batch_size = int(args.batch_size) if args.batch_size > 0 else int(training_cfg.get("batch_size", 8))
    epochs = int(args.epochs) if args.epochs > 0 else int(training_cfg.get("epochs", 50))
    learning_rate = float(args.learning_rate) if args.learning_rate > 0 else float(training_cfg.get("learning_rate", 1e-3))

    tfrecords_dir = (args.tfrecords_dir or Path(env_cfg["paths"]["tfrecords_dir"])).expanduser()
    features_dir = (args.features_dir or Path(env_cfg["paths"]["features_dir"])).expanduser()

    out_root = args.output_root.expanduser() / args.exp_name
    ckpt_dir = out_root / "checkpoints"
    log_dir = out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)

    _install_crepe_stub()
    from ddsp_demucs.data import parse_tfrecord_example
    from ddsp_demucs.model import DDSPDecoder
    from ddsp_demucs.train import DDSPTrainer, setup_training_environment

    train_files = sorted((tfrecords_dir / "train").glob("*.tfrecord"))
    val_files = sorted((tfrecords_dir / "val").glob("*.tfrecord"))
    if args.num_shards_train > 0:
        train_files = train_files[: args.num_shards_train]
    if args.num_shards_val > 0:
        val_files = val_files[: args.num_shards_val]
    if not train_files or not val_files:
        raise RuntimeError(f"Missing TFRecords under {tfrecords_dir}/train or /val")

    feature_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _load_features_np(track_b, start_s, end_s):
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
            feature_cache[track] = (
                data["f0_hz"].astype(np.float32),
                data["loudness_db"].astype(np.float32),
            )

        f0_all, ld_all = feature_cache[track]
        fa = int(round(start_v * frame_rate))
        fb = int(round(end_v * frame_rate))
        if fb <= fa:
            fb = fa + n_frames

        f0 = _fix_len(f0_all[max(0, fa):max(0, fb)], n_frames)
        ld = _fix_len(ld_all[max(0, fa):max(0, fb)], n_frames)
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
        return cond, y

    ds_train = (
        tf.data.TFRecordDataset([str(f) for f in train_files], num_parallel_reads=tf.data.AUTOTUNE)
        .map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_map_to_cond, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(batch_size * 32, seed=args.seed, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_val = (
        tf.data.TFRecordDataset([str(f) for f in val_files], num_parallel_reads=tf.data.AUTOTUNE)
        .map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_map_to_cond, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    setup_training_environment(use_mixed_precision=False, gpu_memory_growth=True, xla_jit=False)

    model = DDSPDecoder(
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        n_harmonics=int(model_cfg.get("n_harmonics", 64)),
        n_noise_bands=int(model_cfg.get("n_noise_bands", 65)),
        rnn_units=int(model_cfg.get("rnn_units", 256)),
        mlp_units=tuple(model_cfg.get("mlp_units", [256, 128])),
        f0_midi_range=tuple(model_cfg.get("f0_midi_range", [24.0, 84.0])),
    )
    trainer = DDSPTrainer(
        model=model,
        sample_rate=sample_rate,
        mel_weight=float(loss_cfg.get("mel_weight", 1.0)),
        hf_mel_weight=float(loss_cfg.get("hf_mel_weight", 0.5)),
        centroid_weight=float(loss_cfg.get("centroid_weight", 0.05)),
        transient_weight=float(loss_cfg.get("transient_weight", 0.2)),
        sisdr_weight=float(loss_cfg.get("sisdr_weight", 0.0)),
    )

    # Build model/trainer state before checkpoint callback serialization.
    sample_cond, _sample_target = next(iter(ds_train.take(1)))
    _ = model(sample_cond, training=False)

    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        run_eagerly=True,  # ddsp.FilteredNoise needs concrete batch dims here.
    )

    class SaveInnerWeights(tf.keras.callbacks.Callback):
        def __init__(self, out_path: Path, monitor: str = "val_val_loss", mode: str = "min"):
            super().__init__()
            self.out_path = out_path
            self.monitor = monitor
            self.mode = mode
            self.best = np.inf if mode == "min" else -np.inf

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if self.monitor not in logs:
                return
            value = float(logs[self.monitor])
            improved = value < self.best if self.mode == "min" else value > self.best
            if improved:
                self.best = value
                self.out_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.model.save_weights(str(self.out_path))

    callbacks = [
        SaveInnerWeights(ckpt_dir / "ddsp.best.weights.h5", monitor="val_val_loss", mode="min"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_val_loss",
            mode="min",
            patience=8,
            restore_best_weights=False,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), write_graph=False, update_freq="epoch"),
    ]

    fit_kwargs = {
        "validation_data": ds_val,
        "epochs": epochs,
        "callbacks": callbacks,
        "verbose": 1,
    }
    if args.train_steps > 0:
        fit_kwargs["steps_per_epoch"] = args.train_steps
    if args.val_steps > 0:
        fit_kwargs["validation_steps"] = args.val_steps

    history = trainer.fit(ds_train, **fit_kwargs)

    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    history_path = out_root / "history.json"
    history_path.write_text(json.dumps(hist, indent=2))

    run_cfg = {
        "exp_name": args.exp_name,
        "sample_rate": sample_rate,
        "frame_rate": frame_rate,
        "win_s": win_s,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "num_train_shards": len(train_files),
        "num_val_shards": len(val_files),
        "train_steps": args.train_steps,
        "val_steps": args.val_steps,
    }
    run_cfg_path = out_root / "run_config.json"
    run_cfg_path.write_text(json.dumps(run_cfg, indent=2))

    print(f"Wrote: {history_path}")
    print(f"Wrote: {run_cfg_path}")
    print(f"Best checkpoint expected at: {ckpt_dir / 'ddsp.best.weights.h5'}")


if __name__ == "__main__":
    main()

