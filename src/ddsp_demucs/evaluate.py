"""Evaluation metrics: SI-SDR, spectral convergence, and plotting utilities."""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio.
    
    Args:
        ref: Reference signal
        est: Estimated signal
        eps: Epsilon for numerical stability
        
    Returns:
        SI-SDR in dB
    """
    ref = ref.astype(np.float64) - np.mean(ref)
    est = est.astype(np.float64) - np.mean(est)
    
    denom = np.dot(ref, ref) + eps
    a = (np.dot(est, ref) / denom) if denom > 0 else 0.0
    e_true = a * ref
    e_res = est - e_true
    
    num = np.sum(e_true**2) + eps
    den = np.sum(e_res**2) + eps
    
    return 10.0 * np.log10(num / den)


def spectral_convergence(a: np.ndarray,
                        b: np.ndarray,
                        n_fft: int = 1024,
                        hop: int = 256,
                        eps: float = 1e-12) -> float:
    """Compute spectral convergence.
    
    Args:
        a: First signal
        b: Second signal
        n_fft: FFT size
        hop: Hop length
        eps: Epsilon
        
    Returns:
        Spectral convergence (lower is better)
    """
    A = np.abs(tf.signal.stft(a, n_fft, hop, n_fft).numpy())
    B = np.abs(tf.signal.stft(b, n_fft, hop, n_fft).numpy())
    return np.linalg.norm(A - B) / (np.linalg.norm(A) + eps)


def evaluate_model(model: tf.keras.Model,
                   dataset: tf.data.Dataset,
                   n_examples: Optional[int] = None,
                   sample_rate: int = 22050) -> Dict[str, List[float]]:
    """Evaluate model on dataset.
    
    Args:
        model: Trained model
        dataset: Evaluation dataset
        n_examples: Number of examples to evaluate (None for all)
        sample_rate: Sample rate for audio
        
    Returns:
        Dictionary with metric lists
    """
    results = {
        "si_sdr": [],
        "spectral_convergence": [],
        "rms": [],
    }
    
    ds = dataset if n_examples is None else dataset.take(n_examples)
    
    for cond, target in ds:
        pred = model(cond, training=False)
        
        # Convert to numpy
        y = target[0].numpy().astype(np.float32)
        p = pred[0].numpy().astype(np.float32)
        
        # Align lengths
        n = min(len(y), len(p))
        y = y[:n]
        p = p[:n]
        
        # Compute metrics
        results["si_sdr"].append(si_sdr(y, p))
        results["spectral_convergence"].append(spectral_convergence(y, p))
        results["rms"].append(np.sqrt(np.mean(p**2) + 1e-12))
    
    return results


def plot_evaluation_results(results: Dict[str, List[float]],
                           output_path: Optional[Path] = None):
    """Plot evaluation results.
    
    Args:
        results: Dictionary of metric lists
        output_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    
    if len(results) == 1:
        axes = [axes]
    
    for ax, (metric, values) in zip(axes, results.items()):
        ax.hist(values, bins=20, alpha=0.7)
        ax.axvline(np.median(values), color='r', linestyle='--', label=f'Median: {np.median(values):.2f}')
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {metric}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def export_evaluation_csv(results: Dict[str, List[float]],
                         output_path: Path,
                         track_names: Optional[List[str]] = None):
    """Export evaluation results to CSV.
    
    Args:
        results: Dictionary of metric lists
        output_path: Path to save CSV
        track_names: Optional list of track names
    """
    df = pd.DataFrame(results)
    
    if track_names is not None:
        df.insert(0, "track", track_names[:len(df)])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Exported evaluation results to {output_path}")


def compare_baseline_vs_model(baseline_audio: np.ndarray,
                              model_audio: np.ndarray,
                              target_audio: np.ndarray) -> Dict[str, float]:
    """Compare baseline vs model performance.
    
    Args:
        baseline_audio: Baseline prediction
        model_audio: Model prediction
        target_audio: Ground truth
        
    Returns:
        Dictionary with delta metrics
    """
    # Align lengths
    n = min(len(baseline_audio), len(model_audio), len(target_audio))
    baseline_audio = baseline_audio[:n]
    model_audio = model_audio[:n]
    target_audio = target_audio[:n]
    
    # Compute metrics
    sdr_baseline = si_sdr(target_audio, baseline_audio)
    sdr_model = si_sdr(target_audio, model_audio)
    sc_baseline = spectral_convergence(target_audio, baseline_audio)
    sc_model = spectral_convergence(target_audio, model_audio)
    
    return {
        "si_sdr_baseline": sdr_baseline,
        "si_sdr_model": sdr_model,
        "delta_si_sdr": sdr_model - sdr_baseline,
        "spectral_convergence_baseline": sc_baseline,
        "spectral_convergence_model": sc_model,
        "delta_spectral_convergence": sc_baseline - sc_model,  # lower is better
    }

