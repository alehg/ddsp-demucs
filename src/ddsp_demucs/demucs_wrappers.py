"""Wrappers for Demucs model inference and audio processing."""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from demucs.pretrained import get_model
from demucs.apply import apply_model


def downmix_mono(x: torch.Tensor) -> torch.Tensor:
    """Downmix stereo to mono by averaging channels.
    
    Args:
        x: Audio tensor of shape (C, T) or (B, C, T)
        
    Returns:
        Mono audio tensor of shape (1, T) or (B, 1, T)
    """
    if x.ndim == 2:
        return x.mean(0, keepdim=True)
    elif x.ndim == 3:
        return x.mean(1, keepdim=True)
    return x


def ensure_sr(x: torch.Tensor, sr_src: int, sr_dst: Optional[int]) -> Tuple[torch.Tensor, int]:
    """Resample audio if target sample rate differs.
    
    Args:
        x: Audio tensor
        sr_src: Source sample rate
        sr_dst: Target sample rate. If None, returns original
        
    Returns:
        Tuple of (resampled audio, sample rate)
    """
    if sr_dst is None or sr_dst == sr_src:
        return x, sr_src
    return torchaudio.functional.resample(x, sr_src, sr_dst), sr_dst


def np_to_torch_wave(np_audio: np.ndarray) -> torch.Tensor:
    """Convert numpy audio array to torch tensor.
    
    musdb gives (T, C) float np array in [-1, 1].
    Demucs expects torch (C, T).
    
    Args:
        np_audio: NumPy array of shape (T, C) or (T,)
        
    Returns:
        Torch tensor of shape (C, T)
    """
    if np_audio.ndim == 1:
        np_audio = np_audio[:, None]
    # (T, C) -> (C, T)
    return torch.from_numpy(np_audio.astype('float32')).permute(1, 0).contiguous()


class DemucsSeparator:
    """Wrapper for Demucs model inference."""
    
    VOCAL_INDEX = 3  # Demucs order: [drums, bass, other, vocals]
    
    def __init__(self, model_name: str = "htdemucs", device: Optional[str] = None):
        """Initialize Demucs separator.
        
        Args:
            model_name: Name of pretrained model
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model = get_model(model_name).to(device).eval()
        self.model_name = model_name
    
    def separate(self, 
                 audio: torch.Tensor,
                 split: bool = True,
                 overlap: float = 0.1) -> torch.Tensor:
        """Separate audio into sources.
        
        Args:
            audio: Input audio tensor of shape (C, T) or (1, C, T)
            split: Whether to split long tracks
            overlap: Overlap ratio for splitting
            
        Returns:
            Sources tensor of shape (nsrc, C, T)
        """
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        
        with torch.inference_mode():
            sources = apply_model(
                self.model,
                audio.to(self.device),
                split=split,
                overlap=overlap
            )[0].cpu()
        
        return sources
    
    def extract_vocals(self,
                       audio: torch.Tensor,
                       split: bool = True,
                       overlap: float = 0.1) -> torch.Tensor:
        """Extract vocals from mixture.
        
        Args:
            audio: Input audio tensor
            split: Whether to split long tracks
            overlap: Overlap ratio for splitting
            
        Returns:
            Vocals tensor of shape (C, T)
        """
        sources = self.separate(audio, split=split, overlap=overlap)
        return sources[self.VOCAL_INDEX]

