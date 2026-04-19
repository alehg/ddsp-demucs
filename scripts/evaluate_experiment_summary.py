#!/usr/bin/env python3
"""Generate evaluation summary artifacts for a trained experiment.

Outputs under results/<exp_name>/evaluation_summary/:
- metrics.csv (per-example rows)
- summary.json (aggregate stats)
- summary.md (human-readable summary)
- histograms.png (metric distributions)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keep TensorFlow logs concise.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
import yaml


def _install_crepe_stub() -> None:
    """Stub crepe to keep ddsp imports working without local crepe install."""
    if "crepe" in sys.modules:
        return
    mod = types.ModuleType("crepe")

    def _predict_stub(*_args, **_kwargs):
        raise RuntimeError("crepe.predict unavailable in this environment")

    mod.predict = _predict_stub
    sys.modules["crepe"] = mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an experiment and export summary artifacts")
    p.add_argument("--exp-name", type=str, required=True, help="Experiment folder name under results/")
    p.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    p.add_argument("--env-config", type=Path, default=Path("env/config.yaml"))
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--model-kind", type=str, default="direct", choices=["direct", "residual"])
    p.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path (defaults to results/<exp>/checkpoints/ddsp.best.weights.h5)",
    )
    p.add_argument(
        "--val-steps",
        type=int,
        default=0,
        help="Override validation steps. If 0, use run_config.json value.",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="If >0, stop after this many examples.",
    )
    return p.parse_args()


def _fix_len(v: np.ndarray, n_frames: int) -> np.ndarray:
    if len(v) >= n_frames:
        return v[:n_frames].astype(np.float32)
    out = np.zeros((n_frames,), dtype=np.float32)
    out[: len(v)] = v.astype(np.float32)
    return out


def _q(series: pd.Series, qv: float) -> float:
    return float(np.quantile(series.to_numpy(), qv))


def main() -> None:
    args = parse_args()
    _install_crepe_stub()

    from ddsp_demucs.data import parse_tfrecord_example
    from ddsp_demucs.evaluate import si_sdr, spectral_convergence
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
        raise RuntimeError("Validation steps not provided and not found in run_config.json")

    features_dir = Path(env_cfg["paths"]["features_dir"]).expanduser()
    tfrecords_dir = Path(env_cfg["paths"]["tfrecords_dir"]).expanduser()
    val_files = sorted((tfrecords_dir / "val").glob("*.tfrecord"))
    if not val_files:
        raise RuntimeError(f"No TFRecords found under {tfrecords_dir}/val")

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

    rows = []
    seen = 0
    for cond, target, track_b, start_sec, end_sec in ds_val:
        pred = model(cond, training=False)
        bsz = int(target.shape[0])
        for i in range(bsz):
            y = target[i].numpy().astype(np.float32)
            p = pred[i].numpy().astype(np.float32)
            xin = cond["x_in"][i].numpy().astype(np.float32)
            n = min(len(y), len(p), len(xin))
            y = y[:n]
            p = p[:n]
            xin = xin[:n]

            sdr_model = float(si_sdr(y, p))
            sdr_base = float(si_sdr(y, xin))
            sc_model = float(spectral_convergence(y, p))
            sc_base = float(spectral_convergence(y, xin))

            track = track_b[i].numpy().decode("utf-8") if hasattr(track_b[i], "numpy") else bytes(track_b[i]).decode("utf-8")
            st = float(start_sec[i].numpy())
            en = float(end_sec[i].numpy())
            rows.append(
                {
                    "track": track,
                    "start_sec": st,
                    "end_sec": en,
                    "si_sdr_model": sdr_model,
                    "si_sdr_baseline": sdr_base,
                    "delta_si_sdr_model_minus_baseline": sdr_model - sdr_base,
                    "spectral_convergence_model": sc_model,
                    "spectral_convergence_baseline": sc_base,
                    "delta_sc_baseline_minus_model": sc_base - sc_model,
                    "rms_model": float(np.sqrt(np.mean(p**2) + 1e-12)),
                    "rms_baseline": float(np.sqrt(np.mean(xin**2) + 1e-12)),
                }
            )
            seen += 1
            if args.max_examples > 0 and seen >= args.max_examples:
                break
        if args.max_examples > 0 and seen >= args.max_examples:
            break

    if not rows:
        raise RuntimeError("No evaluation examples were processed")

    df = pd.DataFrame(rows)
    out_dir = exp_dir / "evaluation_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"
    df.to_csv(metrics_csv, index=False)

    summary = {
        "experiment": args.exp_name,
        "model_kind": args.model_kind,
        "weights_path": str(weights_path),
        "num_examples_evaluated": int(len(df)),
        "protocol": {
            "split": "val",
            "sample_rate": sample_rate,
            "win_s": win_s,
            "frame_rate": frame_rate,
            "batch_size": batch_size,
            "val_steps": val_steps,
            "drop_remainder": True,
        },
        "si_sdr_model": {
            "mean": float(df["si_sdr_model"].mean()),
            "median": float(df["si_sdr_model"].median()),
            "p25": _q(df["si_sdr_model"], 0.25),
            "p75": _q(df["si_sdr_model"], 0.75),
        },
        "si_sdr_baseline": {
            "mean": float(df["si_sdr_baseline"].mean()),
            "median": float(df["si_sdr_baseline"].median()),
            "p25": _q(df["si_sdr_baseline"], 0.25),
            "p75": _q(df["si_sdr_baseline"], 0.75),
        },
        "delta_si_sdr_model_minus_baseline": {
            "mean": float(df["delta_si_sdr_model_minus_baseline"].mean()),
            "median": float(df["delta_si_sdr_model_minus_baseline"].median()),
            "p25": _q(df["delta_si_sdr_model_minus_baseline"], 0.25),
            "p75": _q(df["delta_si_sdr_model_minus_baseline"], 0.75),
        },
        "spectral_convergence_model": {
            "mean": float(df["spectral_convergence_model"].mean()),
            "median": float(df["spectral_convergence_model"].median()),
        },
        "spectral_convergence_baseline": {
            "mean": float(df["spectral_convergence_baseline"].mean()),
            "median": float(df["spectral_convergence_baseline"].median()),
        },
        "delta_sc_baseline_minus_model": {
            "mean": float(df["delta_sc_baseline_minus_model"].mean()),
            "median": float(df["delta_sc_baseline_minus_model"].median()),
        },
    }

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()

    axs[0].hist(df["si_sdr_baseline"], bins=30, alpha=0.7, label="baseline")
    axs[0].hist(df["si_sdr_model"], bins=30, alpha=0.7, label="model")
    axs[0].set_title("SI-SDR distributions")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    axs[1].hist(df["delta_si_sdr_model_minus_baseline"], bins=30, alpha=0.8)
    axs[1].axvline(0.0, color="r", linestyle="--")
    axs[1].set_title("Delta SI-SDR (model - baseline)")
    axs[1].grid(alpha=0.3)

    axs[2].hist(df["spectral_convergence_baseline"], bins=30, alpha=0.7, label="baseline")
    axs[2].hist(df["spectral_convergence_model"], bins=30, alpha=0.7, label="model")
    axs[2].set_title("Spectral Convergence distributions")
    axs[2].legend()
    axs[2].grid(alpha=0.3)

    axs[3].hist(df["delta_sc_baseline_minus_model"], bins=30, alpha=0.8)
    axs[3].axvline(0.0, color="r", linestyle="--")
    axs[3].set_title("Delta SC (baseline - model)")
    axs[3].grid(alpha=0.3)

    plt.tight_layout()
    hist_png = out_dir / "histograms.png"
    fig.savefig(hist_png, dpi=140)
    plt.close(fig)

    summary_md = out_dir / "summary.md"
    summary_md.write_text(
        "\n".join(
            [
                f"# Evaluation Summary - {args.exp_name}",
                "",
                f"- Model kind: `{args.model_kind}`",
                f"- Examples evaluated: {summary['num_examples_evaluated']}",
                f"- SI-SDR median (model): {summary['si_sdr_model']['median']:.4f}",
                f"- SI-SDR median (baseline): {summary['si_sdr_baseline']['median']:.4f}",
                f"- Delta SI-SDR median (model - baseline): {summary['delta_si_sdr_model_minus_baseline']['median']:.4f}",
                f"- Spectral Convergence median (model): {summary['spectral_convergence_model']['median']:.6f}",
                f"- Spectral Convergence median (baseline): {summary['spectral_convergence_baseline']['median']:.6f}",
                f"- Delta SC median (baseline - model): {summary['delta_sc_baseline_minus_model']['median']:.6f}",
                "",
                "Higher SI-SDR is better. Lower Spectral Convergence is better.",
                "",
                f"Artifacts: `{metrics_csv}`, `{summary_json}`, `{hist_png}`",
            ]
        )
    )

    print(json.dumps(
        {
            "metrics_csv": str(metrics_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "histograms_png": str(hist_png),
            "num_examples": int(len(df)),
            "si_sdr_median_model": summary["si_sdr_model"]["median"],
            "si_sdr_median_baseline": summary["si_sdr_baseline"]["median"],
            "delta_si_sdr_median": summary["delta_si_sdr_model_minus_baseline"]["median"],
            "sc_median_model": summary["spectral_convergence_model"]["median"],
            "sc_median_baseline": summary["spectral_convergence_baseline"]["median"],
            "delta_sc_median": summary["delta_sc_baseline_minus_model"]["median"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

