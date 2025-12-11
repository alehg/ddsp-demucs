"""DSP functions: F0 extraction, spectral losses, and audio processing."""

import numpy as np
import torch
import torchcrepe
import tensorflow as tf
import librosa
from typing import Optional, Tuple
import ddsp
from ddsp.losses import SpectralLoss


def torchcrepe_f0(audio_1d: np.ndarray,
                  sr: int,
                  hop_length: int = 512,
                  fmin: float = 50.0,
                  fmax: float = 1100.0,
                  periodicity_thresh: float = 0.45,
                  model_size: str = "full") -> np.ndarray:
    """Extract F0 using torchcrepe.
    
    Args:
        audio_1d: 1D audio array
        sr: Sample rate
        hop_length: Hop length in samples
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz
        periodicity_thresh: Periodicity threshold for voicing
        model_size: Model size ('tiny' or 'full')
        
    Returns:
        F0 array in Hz
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(audio_1d, dtype=torch.float32, device=device)[None]  # [1, T]
    
    with torch.no_grad():
        f0 = torchcrepe.predict(
            x, sr, hop_length,
            torch.tensor([fmin], device=device),
            torch.tensor([fmax], device=device),
            model=model_size,
            batch_size=1024,
            device=device,
            return_periodicity=False
        )[0].cpu().numpy()  # [frames]
    
    # Replace NaNs/Infs
    f0 = np.where(np.isfinite(f0), f0, 0.0)
    return f0


def librosa_f0_pyin(y: np.ndarray,
                     sr: int,
                     fmin: float = librosa.note_to_hz('C2'),
                     fmax: float = librosa.note_to_hz('C7'),
                     frame_length: int = 2048,
                     hop_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract F0 using librosa.pyin.
    
    Args:
        y: Audio array
        sr: Sample rate
        fmin: Minimum frequency
        fmax: Maximum frequency
        frame_length: Frame length for analysis
        hop_length: Hop length (defaults to 0.01 * sr)
        
    Returns:
        Tuple of (f0_hz, voiced_flag, voiced_prob)
    """
    if hop_length is None:
        hop_length = int(0.01 * sr)
    
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length
    )
    
    return np.asarray(f0), np.asarray(voiced_flag), np.asarray(voiced_prob)


def compute_loudness(audio: np.ndarray,
                     sr: int,
                     frame_rate: int = 250,
                     ref_db: float = 20.7) -> np.ndarray:
    """Compute loudness using DDSP.
    
    Args:
        audio: Audio array
        sr: Sample rate
        frame_rate: Frame rate for features
        ref_db: Reference dB level
        
    Returns:
        Loudness array in dB
    """
    audio_tf = tf.expand_dims(tf.constant(audio, dtype=tf.float32), 0)
    ld = ddsp.spectral_ops.compute_loudness(
        audio_tf, sample_rate=sr, frame_rate=frame_rate,
        use_tf=True, ref_db=ref_db
    )
    return tf.squeeze(ld, 0).numpy()


def create_spectral_loss(fft_sizes: Tuple[int, ...] = (2048, 1024, 512, 256, 128, 64),
                         loss_type: str = 'L1',
                         mag_weight: float = 1.0,
                         logmag_weight: float = 0.2,
                         delta_freq_weight: float = 0.2,
                         delta_time_weight: float = 0.05,
                         cumsum_freq_weight: float = 0.0,
                         loudness_weight: float = 0.0) -> SpectralLoss:
    """Create a multi-scale spectral loss.
    
    Args:
        fft_sizes: FFT sizes for multi-scale loss
        loss_type: Loss type ('L1' or 'L2')
        mag_weight: Weight for magnitude loss
        logmag_weight: Weight for log-magnitude loss
        delta_freq_weight: Weight for frequency delta loss
        delta_time_weight: Weight for time delta loss
        cumsum_freq_weight: Weight for frequency cumsum loss
        loudness_weight: Weight for loudness loss
        
    Returns:
        SpectralLoss instance
    """
    return SpectralLoss(
        fft_sizes=fft_sizes,
        loss_type=loss_type,
        mag_weight=mag_weight,
        logmag_weight=logmag_weight,
        delta_time_weight=delta_time_weight,
        delta_freq_weight=delta_freq_weight,
        cumsum_freq_weight=cumsum_freq_weight,
        loudness_weight=loudness_weight,
        name='spectral_loss'
    )


def mel_spectrogram(y: tf.Tensor,
                    n_fft: int = 1024,
                    hop_length: int = 256,
                    n_mels: int = 64,
                    sr: int = 22050,
                    fmin: float = 50.0,
                    fmax: Optional[float] = None) -> tf.Tensor:
    """Compute mel spectrogram.
    
    Args:
        y: Audio tensor of shape [B, T] or [T]
        n_fft: FFT size
        hop_length: Hop length
        n_mels: Number of mel bins
        sr: Sample rate
        fmin: Minimum frequency
        fmax: Maximum frequency (defaults to sr * 0.45)
        
    Returns:
        Mel spectrogram tensor
    """
    if fmax is None:
        fmax = sr * 0.45
    
    if y.shape.rank == 1:
        y = y[tf.newaxis, :]
    
    S = tf.abs(tf.signal.stft(
        y, frame_length=n_fft, frame_step=hop_length, fft_length=n_fft,
        window_fn=tf.signal.hann_window, pad_end=True
    ))  # [B, T, F]
    
    mel_fb = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_fft // 2 + 1,
        sample_rate=sr,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
    )
    
    M = tf.einsum('btf,fm->btm', S, mel_fb)  # [B, T, M]
    return tf.math.log(M + 1e-5)


def spectral_centroid(y: tf.Tensor,
                      n_fft: int = 1024,
                      hop_length: int = 256,
                      sr: int = 22050) -> tf.Tensor:
    """Compute spectral centroid.
    
    Args:
        y: Audio tensor
        n_fft: FFT size
        hop_length: Hop length
        sr: Sample rate
        
    Returns:
        Spectral centroid tensor
    """
    y = tf.cast(y, tf.float32)
    if y.shape.rank == 1:
        y = y[tf.newaxis, :]
    
    S = tf.abs(tf.signal.stft(
        y, frame_length=n_fft, frame_step=hop_length, fft_length=n_fft,
        window_fn=tf.signal.hann_window, pad_end=True
    ))  # [B, T, F]
    
    freqs = tf.linspace(0.0, tf.cast(sr, tf.float32) / 2.0, n_fft // 2 + 1)  # [F]
    num = tf.reduce_sum(S * freqs[tf.newaxis, tf.newaxis, :], axis=-1)  # [B, T]
    den = tf.reduce_sum(S + 1e-8, axis=-1)  # [B, T]
    c = num / (den + 1e-8)  # [B, T]
    return tf.reduce_mean(c, axis=-1)  # [B]

